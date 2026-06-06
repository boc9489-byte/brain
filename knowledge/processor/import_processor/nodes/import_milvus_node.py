from typing import List, Dict, Tuple, Any, Optional, Sequence
from dataclasses import dataclass
from pymilvus import MilvusClient, DataType


from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import EmbeddingError, MilvusError, StateFieldError,FileProcessingError, ValidationError, LLMError
from knowledge.utils.client.storage_clients import StorageClients

@dataclass
class _SCALAR_FIELD_SPC:
    field_name :str
    datatype: DataType
    max_length: Optional[int]=None

    # def __init__(self, field_name, datatype, max_length)

_SCALAR_FIELD:Sequence[_SCALAR_FIELD_SPC] = (
    _SCALAR_FIELD_SPC(field_name = "content", datatype= DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name = "title", datatype= DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name = "parent_title", datatype= DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name = "file_title", datatype= DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name = "item_name", datatype= DataType.VARCHAR, max_length=65535)
)




# 建造者模式
class _MilvusSchmaBuilder():
    """"处理和Milvus字段约束相关的逻辑"""

    @staticmethod
    def build_schema(milvus_client:MilvusClient, dim:int):
        # 动态字段 = 静态字段 + 额外字段
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        
        #1 主键字段
        schema.add_field(field_name= "chunk_id", datatype=DataType.INT64, is_primary = True, auto_id=True)

        # 2 向量字段
        schema.add_field(field_name ="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name ="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 3 标量字段
        # schema.add_field(field_name = "content", datatype=DataType.VARCHAR, max_length=65535)
        # schema.add_field(field_name = "", datatype=DataType.VARCHAR, max_length=65535)
        # schema.add_field(field_name = "", datatype=DataType.VARCHAR, max_length=65535)
        # schema.add_field(field_name = "", datatype=DataType.VARCHAR, max_length=65535)
        # 静态字段
        for spec in _SCALAR_FIELD:
            kwargs: Dict = {
                "field_name":spec.field_name,
                "datatype":spec.datatype,
                # "max_length":spec.max_length
            }
            if spec.max_length:
                kwargs['max_length'] = spec.max_length

            schema.add_field(**kwargs)
        return schema
    
class _MilvusIndexBuilder:
    @staticmethod
    def build_index_params(milvus_client):
        index = milvus_client.prepare_index_params()

        # 稠密向量：AUTOINDEX + COSINE（BGE-M3 已归一化，COSINE ≡ IP）
        index.add_index(field_name="dense_vector", index_name="dense_vector_index",
                        index_type="AUTOINDEX", metric_type="COSINE")

        # 稀疏向量：倒排索引 + 内积（token 权重累加）
        index.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        return index

class _MilvusInserter:
    def __init__(self, milvus_client:MilvusClient, collection_name:str):
        self._milvus_client=milvus_client
        self._collection_name=collection_name

    def insert_rows(self, data:List[Dict[str,Any]]):
        inserted_result = self._milvus_client.insert(collection_name=self._collection_name, data=data)

        chunk_ids = inserted_result.get("ids")

        # for id in chunk_ids:
        for chunk_id,chunk in zip(chunk_ids, data):
            chunk['chunk_id'] = chunk_id
        
        return data


class ImportMilvusNode(BaseNode):
    name = "import_milvus_node"


    def process(self, state: ImportGraphState) -> ImportGraphState:
       
            # 1. 校验state
            validated_chunks,dim = self._validate_state(state)
            # 2. 构建客户端'
            try:
                milvus_client = StorageClients.get_milvus()
            except ConnectionError as e:
                self.logger.error(f"获取milvus 客户端创建失败，异常：{str(e)}")
                raise MilvusError(message =f"获取milvus 客户端创建失败，异常：{str(e)}",node_name = self.name)
            # 3. 获取稠密向量和稀疏向量
            chunks_collection = self.config.chunks_collection
            self._create_chunks_collection(chunks_collection, milvus_client, dim)

            # 4. 处理和Milvus字段约束相关的逻辑
            #     4.1 静态字段 添加额外字段 --> 动态字段
            #     4.2
            _inserter = _MilvusInserter(milvus_client, chunks_collection)

            state['chunks'] = _inserter.insert_rows(validated_chunks)
            return state

    def _create_chunks_collection(self, chunks_collection:str, milvus_client:MilvusClient, dim:int):
        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"{chunks_collection}已存在 跳过创建")
            return
        
        schema = _MilvusSchmaBuilder.build_schema(milvus_client, dim)
        index = _MilvusIndexBuilder.build_index_params(milvus_client)

        milvus_client.create_collection(collection_name=chunks_collection, schema=schema, index_params=index)


    def _validate_state(self, state:ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:
        chunks = state.get('chunks')

        if not chunks or not isinstance(chunks,list):
            raise StateFieldError(node_name=self.name, field_name="chunks",expected_type=List)
        
        validated_chunks = []
        for i,chunk in enumerate(chunks):
            if not chunk or not isinstance(chunk, dict):
                raise ValidationError(message= f"chunks[i] 类型无效：实际为{type(chunk).__name__}",node_name = self.name)
        
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[i] 缺少混合向量，已跳过")
        if not validated_chunks:
            raise ValidationError(message= f"所有chunk均无有效向量，无法入库",node_name = self.name)
        
        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效chunks: {len(validated_chunks)},向量维度 {dim}")

        return validated_chunks,dim
def _cli_main() -> None:

    import  json
    from pathlib import Path
    setup_logging()

    temp_dir = Path(r"knowledge/processor/import_processor/temp_dir"
    )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get('chunks')}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")
if __name__ == "__main__":
    _cli_main()