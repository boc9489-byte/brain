from typing import List, Dict, Any
import re
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import (
    StateFieldError,
    FileProcessingError,
)
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        config = self.config
        # 1.参数校验
        md_content, file_title, max_content_length, min_content_length = (
            self._validat_state(state, config)
        )

        # 2. 根据md 标题切分
        section: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

        # 3.切分或者合并 - langchain 迭代切分器
        final_section = self._split_and_merge(
            section, config.max_content_length, config.min_content_length
        )

        # 4. 组成chunk对象
        final_chunk = self._assemble_chunks(final_section)
        # 5 备份
        self._backup_chunks(state, final_chunk)

        # 6 更新state
        state["chunks"] = final_chunk
        # 返回
        return state

    def _backup_chunks(self, state: ImportGraphState, sections):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

    def _assemble_chunks(self, file_title) -> List[Dict[str, Any]]:
        """
        组装最后的chunks
        Args:
            file_title:

        Returns:

        """
        final_chunk = []
        for section in file_title:
            body = section.get("body")
            title = section.get("title")
            parent_title = section.get("parent_title")
            file_title = section.get("file_title")
            content = f"{title}\n\n{body}"
            final_chunk.append(
                {
                    "content": content,
                    "title": title,
                    "parent_title": parent_title,
                    "file_title": file_title,
                }
            )
        return final_chunk

    def _split_and_merge(
        self,
        sections: List[Dict[str, Any]],
        max_content_length: int,
        min_content_length: int,
    ) -> List[Dict[str, Any]]:
        """
        切分合并
        Args:
            sections: 经过一级标题切分后的章节
            max_content_length: 最大内容长度，保证大部分section 不需要切割
            min_content_length: 相同父标题内容较少，保证小部分的section 需要合并

        Returns:
            先切后合
        """
        current_section = []
        for section in sections:
            current_section.extend(
                self._split_long_section(section, max_content_length)
            )

        final_sections = self._merge_short_section(current_section, min_content_length)

        return final_sections

    def _merge_short_section(self, current_sections, min_content_length):
        """贪心累加算法：将同源的短 section 合并"""
        # 1. 初始化
        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (
                current_section["parent_title"] == next_section["parent_title"]
            )

            if same_parent and len(current_section.get("body")) < min_content_length:
                # 合并 body
                current_section["body"] = (
                    current_section.get("body").rstrip()
                    + "\n\n"
                    + next_section.get("body").lstrip()
                )
                # 标题回退为父标题
                current_section["title"] = current_section["parent_title"]
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section

        # 最后一个封箱
        final_sections.append(current_section)

        return final_sections

    def _split_long_section(
        self, section: Dict[str, Any], max_content_length: int
    ) -> List[Dict[str, Any]]:
        """
        切分
        Args:
            section: 当前章节
            max_content_length: 最大长度阈值

        Returns:
            List[Dict[str, Any]]
        """
        # 1.获取section 对象
        body = section.get("body")
        title = section.get("title")
        parent_title = section.get("parent_title")
        file_title = section.get("file_title")

        if len(title) > 80:
            title = title[:80]

        if "<table>" in body:
            self.logger.info("检查到section中使用到了表格")
            body = MarkdownTableLinearizer().process(body)
            section["body"] = body

        # 2.获取比哦啊题前缀
        title_prefix = f"{title}\n\n"

        # 3.获取总长度
        total_length = len(title_prefix) + len(body)

        # 4.判断总长度是否超过阈值
        if total_length <= max_content_length:
            return [section]

        # 5.切分器对象
        body_length = max_content_length - len(title_prefix)
        if body_length == 0:
            return [section]

        # chunk_s
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "？", "！", "，", "?", "!", ";", " ", ""],
            keep_separator=True,
        )
        sections = text_splitter.split_text(body)

        if len(sections) == 1:
            return [section]

        sub_sections = []
        for index, text in enumerate(sections):
            sub_sections.append(
                {
                    "body": text,
                    "title": f"{title}_{index + 1}",
                    "parent_title": parent_title,
                    "file_title": file_title,
                }
            )

        return sub_sections

    def _split_by_headings(
        self, md_content: str, file_title: str
    ) -> List[Dict[str, Any]]:
        """
        根据markdown标题切分文档
        """
        is_fence = False  # 是否在代码块内
        body_lines = []
        sections = []
        current_title = ""
        hierarchy = [""] * 7  # 标题层级关系栈
        current_level = 0

        def _flush() -> List[Dict[str, Any]]:
            """ "
            收集：1. 标题
                 2. 行信息
                 3. 父标题
                 4. 文档标题
            """
            body = "\n".join(body_lines).strip()
            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append(
                    {
                        "body": body,
                        "title": current_title if current_title else file_title,
                        "parent_title": parent_title,
                        "file_title": file_title,
                    }
                )

        md_lines = md_content.split("\n")

        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)$")

        for md_line in md_lines:
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                # 遇到代码块，跳过
                is_fence = not is_fence

            match = heading_re.match(md_line) if not is_fence else None

            if match:
                # 将 body_lines 中手机到的行合并成一个 section
                _flush()

                current_title = md_line.strip()

                level = len(match.group(1))
                current_level = level
                hierarchy[level] = current_title

                # 清空
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
                body_lines = []
            else:
                body_lines.append(md_line)
        _flush()
        return sections

    def _validat_state(self, state: ImportGraphState, config) -> ImportGraphState:

        # 步骤日志
        self.log_step("step1", "validating state")

        # 获取md_content
        md_content = state.get("md_content")

        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 获取文档标题
        file_title = state.get("file_title")

        # 判断评估 chunk_size
        if (
            config.max_content_length <= 0
            or config.min_content_length <= 0
            or config.max_content_length < config.min_content_length
        ):
            raise ValueError(
                f"Invalid chunk size: max_content_length={config.max_content_length}, min_content_length={config.min_content_length}"
            )

        return (
            md_content,
            file_title,
            config.max_content_length,
            config.min_content_length,
        )


if __name__ == "__main__":
    setup_logging()

    document_split_node = DocumentSplitNode()

    md_path = r"knowledge/processor/import_processor/万用表RS-12的使用/hybrid_auto/万用表RS-12的使用.md"

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    init_state = {
        "md_content": md_content,
        "file_title": "万用表的使用",
        "file_dir": r"knowledge/processor/import_processor/temp_dir",
    }
    document_split_node.process(init_state)
