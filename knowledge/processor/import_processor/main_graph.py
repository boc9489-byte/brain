import json
from langgraph.graph.state import END, CompiledStateGraph, StateGraph
# from langgraph.graph import 

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import Pdf2MdNode
from knowledge.processor.import_processor.nodes.md_img_node import MarkDownToImgNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode

def import_router(state: ImportGraphState):
    """根据state中 决定去哪个节点"""
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    if state.get("is_md_read_enabled"):
        return "md_img_node"

def import_graph()->CompiledStateGraph:
    """职责：
    1.定义运行时的图状态
    2.定义节点
    3.定义边
    4.返回状态"""

    work_flow = StateGraph(ImportGraphState) #type:ignore

    #2.定义节点
    work_flow.set_entry_point("entry_node")
    node_name_obj= {
        "entry_node": EntryNode(),
        "pdf_to_md_node": Pdf2MdNode(),
        "md_img_node": MarkDownToImgNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "import_milvus_node": ImportMilvusNode(),
        "embedding_chunks_node": EmbeddingChunksNode(),
    }

    for node_name,node_obj in node_name_obj.items():
        work_flow.add_node(node_name,node_obj)

    #定义边
    work_flow.add_conditional_edges("entry_node", import_router, {
        # "entry_node": EntryNode(),
        "pdf_to_md_node":"pdf_to_md_node", #   key : import_router return
        "md_img_node":"md_img_node",
        END:END
    })


    work_flow.add_edge("pdf_to_md_node","md_img_node")
    work_flow.add_edge("md_img_node","document_split_node")
    work_flow.add_edge("document_split_node","item_name_recognition_node")
    work_flow.add_edge("item_name_recognition_node","embedding_chunks_node")
    work_flow.add_edge("embedding_chunks_node","import_milvus_node")
    work_flow.add_edge("import_milvus_node",END)

    builder = work_flow.compile()
    return builder

import_app = import_graph()


# test
def run_import_graph():
    # import_graph()
    graph_state = {
        "import_file_path": r"knowledge/processor/import_processor/temp_dir/万用表RS-12的使用.pdf",
        "file_dir":r"knowledge/processor/import_processor/temp_dir"
    }
    for event in import_app.stream(graph_state):
        final_state= {}
        for key, value in event.items():
            print(f"当前节点{key}")
            print(f"当前节点状态{value}") 
            final_state = value
    return final_state

if __name__ == "__main__":

    setup_logging()
    final_state = run_import_graph()

    print(json.dumps(final_state, ensure_ascii=False, indent=4))

    print(import_app.get_graph().print_ascii())

