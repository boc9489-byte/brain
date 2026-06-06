
import base64
from collections import deque
from dataclasses import dataclass
from logging import Logger
import re
import select
import time
from typing import Deque, List, Dict, Optional, Tuple, Set
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError,FileProcessingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients

@dataclass
class ImageContext:
    """
    图片上下文信息

    """
    head: str  # 上文标题内容
    pre_text:str # 上文内容
    post_text:str # 下文内容

@dataclass
class ImageInfo:
    """
    图片的完整信息
    """
    image_context: ImageContext # 上下文信息
    name: str                   # 图片名称
    path: str                  # 图片地址

class MarkDownToImgNode(BaseNode):
    """
    分别调用四个类方法：
    1.得到四个类的实例对象
    2.分别调用四个实例对象的处理方法
    

    """
    def __init__(self):
        super().__init__()
        self._md_file_handler = _MdFileHandler(self.logger,self.name)
        self._img_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer =_VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._img_uploader =_ImageUploader(self.logger)


    name = "md_to_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑：

        """
        config = self.config
        # 1. 操作 _md_file_handler
        self.log_step("setp1","读取md内容，路径以及图片的目录")
        md_content, md_path_obj, img_dir_obj = self._md_file_handler.validate_and_read_md(state)

        if not img_dir_obj.exists():
            state["md_content"] = md_content
            return state 
 
        # 2. 操作 _img_scanner
        self.log_step("step2","准备开始扫描图片目录")
        img_info_list:List[ImageInfo] = self._img_scanner.scan_imgs_dir(img_dir_obj, 
                                        md_content,
                                        config.image_extensions, 
                                        # 图片上下文的长度
                                        config.img_content_length)
        # 操作 _VLMSummarizer
        self.log_step("step3","通过VLM模型提炼摘要信息")
        summaries:Dict[str,str] = self._vlm_summarizer._summary_all(md_path_obj.stem, img_info_list, config.vl_model)

        # 操作 _img_uploader
        self.log_step("step4","上传图片信息")
        new_md_content = self._img_uploader.upload_and_replace(md_content,img_info_list, md_path_obj.stem, summaries, config.get_minio_base_url(), config.minio_bucket)
        
        self.log_step("step_5","备份新文件")
        self._md_file_handler.backup(md_path_obj, new_md_content)

        state["md_content"] = new_md_content
        return state

class _MdFileHandler:
    """
    职责：
    1.读取 md内容，路径 和 图片目录
    2.备份新的md_content

    """
    ImageContext(
        head="",
        pre_text="",
        post_text=""
    )
    def __init__(self, logger:Logger, node_name:str) -> None:
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state) -> Tuple[str, Path, Path]:
        """
        读取md 内容，路径，和 图片目录
        """
        # 1.从state 中获取md_path
        md_path = state.get("md_path",'')

        # 2.非空判断
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_path",expected_type=str)
        
        # 3. path 标准话
        md_path_obj = Path(md_path)

        # 4.判断路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name="md_path",expected_type=str)

        # 5.读取md_content
        try:
            with open(md_path_obj,"r",encoding="utf-8") as f:
                md_content = f.read()
        except IOError as e:
            self.logger.error(f"{md_path_obj.name}MD文件打开失败")
            raise FileProcessingError(message="文件打开失败",node_name=self.node_name)

        # 6.获取图片目录
        img_dir = md_path_obj.parent / "images"

        # 7.return
        return md_content, md_path_obj, img_dir 
    
    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        """
        备份新的md_content
        """

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)
        

class _VLMSummarizer:
    """
    职责：
    根据图片信息以及图片的上下文信息，生成图片的摘要信息
    """
    def __init__(self, logger:Logger, requests_per_minute:int) -> None:
        self.logger = logger
        self.requests_per_minute = requests_per_minute
        # self.node_name = node_name
    def _summary_all(self, document_name:str, img_info_list:List[ImageInfo], vl_model:str) ->Dict[str,str]:
        """为所有图片生成摘要"""
        summaries = {}
        request_timestaps: Deque[float] = deque()
        # 获取 vlm 客户段
        try:
            vlm_client = AIClients.get_openai()
        except Exception as e:
            # 给图片默认值
            for img_info in img_info_list:
                summaries[img_info.name] = "暂无图片摘要"
            return summaries
        
        # 为每张图片生成摘要
        for img_info in img_info_list:
            self._enforce_rate_limit(request_timestaps, self.requests_per_minute, 60)
            summaries[img_info.name] = self._summary_one(document_name, img_info, vlm_client, vl_model)
        
        self.logger.info(f"生成图片摘要{len(summaries)} 个")
        return summaries

    def _summary_one(self, document_name:str, img_info:ImageInfo, vlm_client:OpenAI, vl_model:str) ->str:
        """为当前图片生成摘要信息"""
        parts = [p for p in (img_info.image_context.head, img_info.image_context.post_text, img_info.image_context.pre_text) if p]

        final_context = "\n".join(parts) if parts else "暂无上下文"

        try:
            with open(img_info.path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
        except IOError as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"

        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {img_info.path}: {e}")
            return "暂无图片描述"
    
    def _enforce_rate_limit(
        self, timestamps: Deque[float],
        max_requests: int, window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)

class _ImageUploader:
    """
    职责：
    1. 将本地图片上传到MinIO,
    2. 替换md 的摘要和地址
    """
    def __init__(self, logger:Logger) -> None:
        self.logger = logger
        # self.node_name = node_name
    def upload_and_replace(self, md_content:str , img_info_list:List[ImageInfo] , object_dir_name:str, summaries:Dict[str,str], minio_url:str, minio_bucket_name:str) ->str:
        """上传图片和摘要"""
        # remote_urls = {}
        # # 获取minio 客户端
        # try:
        #     minio_client = ""
        # except Exception as e:
        #     for img_info in img_info_list:
        #         remote_urls[img_info.name] = img_info.path

        # # 上传一个图片
        # for img_info in img_info_list:
        #     self._upload_one(object_dir_name, minio_url, img_info, minio_bucket_name)
        # pass
        # 上传
        remote_urls = self._upload_all(object_dir_name, img_info_list, minio_url, minio_bucket_name)

        md_content = self._update_md(md_content, summaries, remote_urls)

        return md_content
        
    def _upload_all(self, object_dir_name:str, img_info_list:List[ImageInfo], minio_url:str, minio_bucket_name:str) -> Dict[str,str]:
        remote_url = {}
        try:
            minio_client = StorageClients.get_minio()
        except Exception as e:
            for img_info in img_info_list:
                remote_url[img_info.name] = img_info.path
            return remote_url

        # 上传图片
        for img_info in img_info_list:
            obj_name = f"{object_dir_name}/{img_info.name}"
            try:
                minio_client.fput_object(
                    minio_bucket_name, obj_name, img_info.path
                )
                self.logger.info(f"远程图片地址上传成功 {img_info.name}")
                remote_url[img_info.name] = f"{minio_url}/{minio_bucket_name}/{obj_name}"
                print(f"{remote_url[img_info.name] = }")
            except Exception as e:
                self.logger.warn(f"远程图片地址上传失败 {img_info.name}")
                remote_url[img_info.name] = img_info.path
        self.logger.info(f"获取远程的{len(remote_url)}图片地址")
        return remote_url
    
    def _update_md(self, md_content:str, summaries:Dict[str,str], remote_urls:Dict[str,str]) -> str:
        """更新图片"""
        pattern = re.compile(f"!\[(.*?)\]\((.*?)\)")

        def replacer(match:re.match):
            """替换摘要，minio图片地址"""

            for img_name,img_summary in summaries.items():
                img_path = match.group(2)
                img_name_in_md = Path(img_path).name

                if img_name_in_md == img_name:
                    return f"![{img_summary}]({remote_urls[img_name]})"
            return match.group(0)
        return pattern.sub(replacer,md_content)

class _ImageScanner:
    """
    职责：
    1。根据图片目录，得到有效图片文件
    2. 定位图片的位置
    3. 获取图片在md 中的上下文内容 给 VLM 模型提供上下文
    4. 组装所有图片的上下文内容

    
    """
    def __init__(self, logger:Logger) -> None:
        self.logger = logger
        # self.node_name = node_name

    def scan_imgs_dir(self, img_dir_obj:Path,  md_content:str,
                                        image_extensions:Set[str], 
                                        img_content_length:int) -> List[ImageInfo]:
        """
        1.扫描所有图片文件
        2.获取每个图片文件位置
        2.1 上文信息 标题 + 上文内容
        2.2 下文信息 下文内容
        3.将图片上下文放到 图片信息中
        4. return
        """
        # 1.遍历图片目录
        img_info_list = []
        for img_path in img_dir_obj.iterdir():
            # 1.1 过滤子目录
            if not img_path.is_file():
                self.logger.error(f"{img_path} 不是一个有效的文件")
                continue
            # 1.2 后缀校验
            if  not img_path.suffix in image_extensions:
                self.logger.error(f"{img_path.suffix}不支持的后缀格式")
                continue
            # 1.3 获取图片上下文
            ctx = self._find_context(img_path.name, md_content, img_content_length)

            if not ctx:
                self.logger.error(f"{img_path.name} 未找到引用")
                continue
            # 1.4 封装 imageIno 容器中
            img_info_list.append(ImageInfo(
                name = img_path.name,
                path = str(img_path),
                image_context=ctx
            ))

        self.logger.info(f"找到图片引用 {len(img_info_list)} 个")
        return img_info_list
    
    def _find_context(self, img_name:str, md_content:str, img_content_length:int) -> Optional[ImageContext]:
        """MD 图片文件获取  () 获取组信息 """
        pattern = re.compile(r"!\[(.*?)\]\(.*?"+ re.escape(img_name) + r".*?\)")

        md_lines = md_content.split("\n")
        
        for md_idx, md_line in enumerate(md_lines):
            if not pattern.search(md_line):
                continue

            head, prev_index = self._find_heading_up(md_lines, md_idx)
            pre_lines = md_lines[prev_index + 1: md_idx]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction = "front")


            next_index = self._find_heading_down(md_lines, md_idx)
            next_lines = md_lines[md_idx + 1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length, direction = "back")
            
            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context
            )
        
        return None

    def _extract_limited_context(self, extracted_md_lines:List[str], img_content_length:int, direction:str) ->str:
        """截取上下文内容,保证内容完整"""
        current_paramgraph = []
        paramgraphs = []
        for line in extracted_md_lines:
            is_blank_line = not line.strip()

            is_other_img = re.match(
                r"^!\[.*?\]\(.*?\)$", line.strip()
            )

            if is_blank_line or is_other_img :
                if current_paramgraph:
                    paramgraphs.append("\n".join(current_paramgraph))
                    current_paramgraph = []
                continue
            current_paramgraph.append(line)

        # 处理空行后的行
        if current_paramgraph:
                    paramgraphs.append("\n".join(current_paramgraph))

        if direction == "front":
            paramgraphs.reverse()

        # 保留数据
        total = 0
        selected = []
        for paramgraph in paramgraphs:
            if total + len(paramgraph) > img_content_length and selected:
                break
            selected.append(paramgraph)
            total += len(paramgraph)

        if direction == "front":
            selected.reverse()
        # 转换str
        return "\n\n".join(selected)
        

    def _find_heading_up(self, md_lines:List[str], from_idx: int) -> Tuple[str,int]:
        """获取上文内容"""
        for i in range(from_idx -1,-1,-1):
            if re.match(r"^#{1,6}\s+",md_lines[i]):
                return md_lines[i], i
        
        return "",-1

    def _find_heading_down(self, md_lines:List[str], from_idx: int) -> Tuple[str,int]:
        """获取下文内容"""
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+",md_lines[i]):
                return i
        
        return len(md_lines)

        
if __name__ == "__main__":

    setup_logging()
    md_img_node = MarkDownToImgNode()
    init_state ={
        "md_path":"knowledge/processor/import_processor/万用表RS-12的使用/hybrid_auto/万用表RS-12的使用.md"
    }
    md_img_node.process(init_state)
