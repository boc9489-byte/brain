from typing import List, Dict, Any
import json
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import EmbeddingError, StateFieldError,FileProcessingError, ValidationError, LLMError
from knowledge.utils.client.ai_clients import AIClients  
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunk_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.log_step("step1","校验chunks的数据结构")
        validated_chunks = self._validate_state(state)

        self.log_step("step2","获取BGE-M3嵌入模型客户端")

        try:
            embed_model = AIClients.get_bge_m3_openai()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败，原因：{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败，原因：{str(e)}", node_name=self.name)

        self.log_step("step3","嵌入模型")
        batch_size = self.config.embedding_batch_size

        total = len(validated_chunks)

        final_chunks =[]
        for index in range(0,total,batch_size):
            batch_chunks = validated_chunks[index:index+batch_size]

            batch_end = index + len(batch_chunks)
            self.logger.info(f"嵌入批次 【{index +1} - {batch_end}】/ {total}")

            # 嵌入
            current_chunks = self._embed_chunks(embed_model, batch_chunks)
            final_chunks.extend(current_chunks)

        state["chunks"] = final_chunks
        return state
        
    def _embed_chunks(self, embed_model:BGEM3EmbeddingFunction, batch_chunks:List[Dict[str, Any]]) ->List[Dict[str, Any]]:
        
        embedding_documents:List = [ f"{chunk.get('item_name','')}\n{chunk.get('content')}" for chunk in batch_chunks]
        try:
            embed_vector = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入失败：{str(e)}",node_name = self.name)
        
        if not embed_vector:
            raise EmbeddingError(message=f"嵌入失败,结果不存在：{embed_vector}",node_name = self.name)

        sparse_csr=embed_vector.get('sparse')
        for i,chunk in enumerate(batch_chunks):
            chunk["dense_vector"] = embed_vector.get("dense")[i].tolist()
            chunk["sparse_vector"] = self._extract_sparse_vector(sparse_csr, i)

        return batch_chunks

    def _extract_sparse_vector(self, sparse_csr, index:int):
        
        # sparse_csr = vector_result.get('sparse')
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index+1]

        token_id = sparse_csr.indices[start_index:end_index].tolist()
        weight = sparse_csr.data[start_index:end_index].tolist()

        return dict(zip(token_id, weight))

    def _validate_state(self, state:ImportGraphState) -> List[Dict[str, Any]]:
        chunks = state.get("chunks")
        print(f"{type(chunks) }")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name =self.name, field_name="chunks", expected_type=List)

        for index, chunk in enumerate(chunks):
            # 
            if not isinstance(chunk, dict):
                raise ValidationError(message=f"[chunks_{index + 1}]类型出错，类型{type(chunk)}",
                    node_name =self.name
                )

        return chunks

if __name__ == '__main__':
    setup_logging()

    base_dir = Path(
        r"knowledge/processor/import_processor/temp_dir"
    )
    input_path = base_dir / "chunks_vector_item.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data.get('chunks')})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")
