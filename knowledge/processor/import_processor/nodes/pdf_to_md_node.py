
import json
import subprocess
import time
from typing import Tuple
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode,setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import PdfConversionError, StateFieldError
class Pdf2MdNode(BaseNode):

    name = "pdf_to_md_node"
    # crul + o 
    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 获取 -p 和  -ō
        import_file_path_obj, file_dir_obj = self._vallidate_state(state)

        # 执行 mineru 解析 ( mineru -p input_path -o output_path --source=local)
        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)

        if processed_code:
            raise PdfConversionError(message="mineru 解析失败", node_name=self.name)

        md_path = self.get_md_path(import_file_path_obj, file_dir_obj)

        state["md_path"] = md_path
        return state

    def get_md_path(self, import_file_path_obj:Path, file_dir_obj:Path) -> str:
        file_name = import_file_path_obj.stem
        return str(file_dir_obj / file_name / "hybrid_auto" / f"{file_name}.md")

    def _execute_mineru_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        self.log_step("step2","执行子进程 cmd : mineru -p input_path -o output_path --source=local ")
        
        cmd =[
            "mineru",
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
            "--source",
            "local"
        ]

        # 通过管道 获取子进程正常日志和错误日志
        start_time = time.time()
        proc = subprocess.Popen(args=cmd, 
                        # 接收正常日志
                        stdout= subprocess.PIPE,
                        # 接收错误日志
                        stderr= subprocess.STDOUT,
                        # 输出为文本， 默认输出二进制流
                        text=True,
                        # 替换特殊字符 避免报错
                        errors="replace",
                        encoding="utf-8",
                        # 按行输出 
                        bufsize=1
                        )

        for line in proc.stdout:
            self.logger.info(f"minerU解析产生的日志:{line}")

        # 主线程等待
        processd_rusult = proc.wait()

        end_time = time.time()
        if processd_rusult == 0 :
            self.logger.info(f"mineru 解析完成 , {end_time - start_time : 4f}")
        else:
            self.logger.info("minerU 解析错误")
        
        return processd_rusult

    def _vallidate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:

        # 获取解析的文件path
        self.log_step("step1","准备和获取文件路径和输出目录")
        import_file_path = state.get("import_file_path", '')

        if not import_file_path :
            raise StateFieldError(node_name=self.name, field_name = "import_file_path", expected_type = str)

        import_file_path_obj = Path(import_file_path)
        # print(import_file_path_obj,"====")
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name =self.name , field_name = "import_file_path", 
                expected_type = str, message = "解析的文件路径不存在")

        file_dir = state.get("file_dir", "")

        if not file_dir:
            file_dir = import_file_path_obj.parent
        
        file_dir_obj = Path(file_dir)

        if not file_dir_obj.exists():
            raise StateFieldError(node_name =self.name , field_name = "import_file_path", 
                expected_type = str, message = "输出目录不存在")

        self.logger.info(f"解析的文件路径 {import_file_path_obj} ")
        self.logger.info(f"输出的文件目录 {file_dir_obj} ")
        return import_file_path_obj, file_dir_obj 

if __name__ == "__main__":
    setup_logging()
    pdf_to_md_node = Pdf2MdNode()

    init_data={
        "import_file_path": r"knowledge/processor/import_processor/万用表RS-12的使用.pdf",
        "file_dir": r"knowledge/processor/import_processor"
    }

    result = pdf_to_md_node.process(init_data)
    result_str = json.dumps(result, indent=4, ensure_ascii=False)
    print(result_str)