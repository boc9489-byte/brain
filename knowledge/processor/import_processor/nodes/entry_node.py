from pathlib import Path

# from pydantic import ValidationError
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError,ValidationError,FileProcessingError


class EntryNode(BaseNode):

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        
        import_file_path = state.get("import_file_path",'')
        file_dir = state.get("file_dir",'')

        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)
        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str)
        
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        # if not import_file_path_obj.exists():
        #     raise StateFieldError(node_name=self.name, field_name="import_file_path_obj", expected_type=Path)
        # if not file_dir_obj.exists():
        #     raise StateFieldError(node_name=self.name, field_name="file_dir_obj", expected_type=Path)
        print(import_file_path_obj)
        print(file_dir_obj.exists())
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='import_file_path_obj', expected_type=Path)
        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name='file_dir_obj', expected_type=Path)

        if import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = str(import_file_path_obj)
        else:
            self.logger.error(f"该文件后缀格式{import_file_path_obj.suffix}不支持")
            raise ValidationError(message=f"该文件的后缀格式{import_file_path_obj.suffix}不支持", node_name=self.name)

        state["file_title"] = import_file_path_obj.stem

        return state

if __name__ == "__main__":
    setup_logging()
    entry_node = EntryNode()
    init_state = {
        # "import_file_path":"knowledge/processor/import_processor/万用表RS-12的使用/hybrid_auto/万用表RS-12的使用.md",
        "import_file_path":r"knowledge/processor/import_processor/万用表RS-12的使用.pdf",
        "file_dir":r"knowledge/processor/import_processor"
    }
    result = entry_node.process(init_state)
    import json
    print(json.dumps(result,indent=4, ensure_ascii=False))