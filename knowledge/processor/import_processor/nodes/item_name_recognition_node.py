# from ast import Dict
from multiprocessing import context
from re import S
from typing import List, Optional, Tuple, Any, Dict

from langchain_openai import ChatOpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import DataType, MilvusClient

from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError,FileProcessingError, ValidationError, LLMError, EmbeddingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.prompt.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE
from knowledge.utils.client.storage_clients import StorageClients

class ItemNameRecognitionNode(BaseNode):

    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        1.使用LLM提取商品型号
        2.嵌入商品型号
        3.存储到Milvus
        Args:
            state:

        Returns:

        """
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)

        # 构建上下文
        item_name_context = self._prepare_llm_context(chunks, item_name_chunk_k)

        # 提取LLM
        item_name = self._recognition_item_name(item_name_context, file_title)

        # 向量化
        dense_vector, sparse_vector = self._embedding_item_name(item_name)

        # 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)

        self._fill_item_name(state, chunks, item_name)
        return state

    def _fill_item_name(self, state:ImportGraphState, chunks:List[Dict], item_name:str):
        """回填 item给 chunk， state"""
        chunks = state.get("chunks")
        for chunk in chunks:
            chunk["item_name"] = item_name
        
        state["item_name"] = item_name

    def _insert_milvus(self, dense_vector:List, sparse_vector:Dict[str, Any], file_title:str, item_name:str):
        """写入milvus中"""
        if not dense_vector or not sparse_vector:
            return
        try:
            milvus_client = StorageClients.get_milvus()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败：{str(e)}")
            return 
        
        # 集合：集合名，约束， 索引
        item_name_collection_name = self.config.item_name_collection

        try:
            # 幂等性校验
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)

            # 构建数据
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            # 插入数据
            inserted_result = milvus_client.insert(collection_name=item_name_collection_name, data=[item_name_data_row])
        except Exception as e:
            self.logger.error(f"商品名:{item_name}插入失败 {str(e)}")
            return
        self.logger.info(f"插入的结果:{inserted_result},主键:{inserted_result.get('ids')}")


    def _create_item_name_collection(self, item_name_collection_name:str, milvus_client:MilvusClient):
        
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True,max_length=10)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)

        schema.add_field(field_name= "dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name= "sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 主键索引， 标量索引， 向量索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name = "dense_vector",
            index_name = "dense_vector_index",
            index_type = "AUTOINDEX",
            metric_type= "COSINE"
        )
        index_params.add_index(
            field_name = "sparse_vector",
            index_name = "sparse_vector_index",
            index_type = "SPARSE_INVERTED_INDEX",
            metric_type= "IP" #BM25
        )


        
        milvus_client.create_collection(
                collection_name=item_name_collection_name,
                schema=schema,
                index_params=index_params
        )

        self.logger.info(f"创建{item_name_collection_name}成功")

    def _embedding_item_name(self, item_name:str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        try:
            bge_m3_client: BGEM3EmbeddingFunction = AIClients.get_bge_m3_openai()
        except ConnectionError as e:
            self.logger.error(f"BGE_M3嵌入模型客户端创建失败：{str(e)}")
            raise EmbeddingError(
                message=f"BGE_M3嵌入模型客户端创建失败：{str(e)}",
                node_name=self.name,
                cause=e,
            ) from e
        
        try:
            vector_result = bge_m3_client.encode_documents(documents=[item_name])

            vector_dense = vector_result.get('dense')[0].tolist()

            sparse_csr = vector_result.get('sparse')
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            weight = sparse_csr.data[start_index:end_index].tolist()

            sparse_vector = dict(zip(token_id, weight))
            self.logger.info(f"BGE_M3嵌入模型稠密向量维度：{len(vector_dense)}")
            return vector_dense,sparse_vector
        except Exception as e:
            self.logger.error(f"BGE_M3嵌入模型计算失败：{item_name} 原因：{str(e)}")
            raise EmbeddingError(
                message=f"BGE_M3嵌入模型计算失败：{item_name} 原因：{str(e)}",
                node_name=self.name,
                cause=e,
            ) from e


    def _recognition_item_name(self, item_name_context: str, file_title:str):
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM 客户端创建失败: {str(e)}")
            raise LLMError(message=f"OpenAI 的LLM 客户端创建失败: {str(e)}")

        system_prompt = ITEM_NAME_SYSTEM_PROMPT
        
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                                file_title = file_title,
                                context = item_name_context
                       )
        try:
            llm_response = llm_client.invoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ])
            llm_result = llm_response.content.strip('')
            # print(f"{llm_response = }")
            if not llm_result or llm_result == "UNKNOWN":
                self.logger.error(f"LLM提取商品信息失败, 使用文件名:{file_title}作为商品名")
                return file_title
            self.logger.info(f"LLM为文档：{file_title} 提取商品名： {llm_result}")

            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品信息失败, 使用文件名:{file_title}作为商品名")
            return file_title
        

    def _prepare_llm_context(self, chunks: List[Dict], item_name_chunk_k:int) -> str:

        final_context = []
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            if not isinstance(chunk, dict):
                continue 
        
            content = chunk.get("content")

            splice_context = f"【切片]】 - f{index} - {content}"

            final_context.append(splice_context)
        
        return "\n".join(final_context)


    def _validate_state(self, state) -> Tuple[str, List, int, int]:

        file_title = state.get("file_title")

        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title",expected_type=str)

        chunks = state.get("chunks")

        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="file_title",expected_type=str)

        item_name_chunk_k = self.config.item_name_chunk_k
        print(f"{type(item_name_chunk_k)}")

        item_name_chunk_size = self.config.item_name_chunk_size

        if  not item_name_chunk_k  or item_name_chunk_k <= 0:
            raise ValidationError("商品识别的辅助切片数不合法")

        if not item_name_chunk_size  or item_name_chunk_size <= 0:
            raise ValidationError("商品识别的辅助切长度不合法")
        
        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

import json
from knowledge.processor.import_processor.base import setup_logging

if __name__ == '__main__':
    from pathlib import Path
    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r"knowledge/processor/import_processor/temp_dir/chunks.json"
    base_dir = Path(
        r"knowledge/processor/import_processor/temp_dir"
    )
    output_path = base_dir / "chunks_vector_item.json"


    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")


    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result,f, ensure_ascii=False, indent=4)
    
    print(f"item_name: {result.get('item_name')} 生成完成，结果保存至：\n{output_path}")
    
