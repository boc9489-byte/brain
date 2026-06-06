from typing import List, Dict, Any, Tuple

from langchain_core.runnables import history
from langchain_openai import ChatOpenAI
from transformers.models.tapas.tokenization_tapas import Question

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.prompt.query_prompy import ANSWER_PROMPT

class AnswerOutPutNode(BaseNode):
    def process(self, state: T) -> T:

        """
        核心逻辑：
        1. 从state中获取answer
        1.1 如果获取到了answer:--->没有进行三路检索[不用在生成答案，直接返回]-----【答案如何推送给前端：1.流式（直接将已经生成的内容都给前端）2.非流式（直接将已经生成的内容都给前端）】
        1.2 如果没有获取answer---->进行了三路检索[需要调用LLM生成答案，在返回]----【答案如何推送给前端：1.流式推送（SSE）  2.非流式的(传统)】明显变化
        Args:
            state:
        Returns:
        """
        is_stream = state.get('is_stream')

        task_id = state.get('task_id')

        if state.get('answer'):
            self._push_exist_answer(task_id, is_stream, state)
            is_streamed = False 
        else:
            prompt = self._build_prompt(state)
            state['prompt'] = prompt

            self._generate_answer(prompt, task_id, state)
            is_streamed = is_stream
        
        if is_stream:
            if is_streamed:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
            else:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": state.get('answer')})

        def _generate_answer(prompt, task_id, state):
            """
            调用LLM  生成答案 更新到state
            Args:
                prompt:  提示词
                task_id: 任务id
                state:
            Returns:
            """
            try:
                llm_client = AIClients.get_llm_client(response_format=False)
            except ConnectionError as e:
                self.logger.error(f"获取LLM客户端失败 原因： {str(e)}")

            
            if state.get('is_stream'):
                state['answer'] = self._stream_llm(task_id, prompt, llm_client)

            else:
                state['answer'] = self._invoke_llm(prompt, llm_client)
                set_task_result(task_id=task_id, key="answer", value=state['answer'])

        def _invoke_llm(self, prompt, llm_client: ChatOpenAI):
            try:
                llm_res = llm_client.invoke(prompt)

                llm_content = getattr(llm_res, 'content', '') or ''

                if not llm_content:
                    return "LLM暂无回答"

                return llm_content
            except Exception as e:
                return "LLM暂无回答"

        def _invoke_llm(self, prompt, llm_client):
            accelerate_delta = ""
            try:
                for chunk in llm_client.stram(prompt):
                    delta_text = getattr(chunk, 'content', "") or ""

                    if delta_text:
                        push_sse_event(task_id=task_id,
                                    event=SSEEvent.DELTA,
                                    data={"delta":delta_text})
                        accelerate_delta += delta_text
            except Exception as e:
                return "LLM暂无回答"

            return accelerate_delta

        def _push_exist_answer(self, task_id, is_stream, state):
            if not is_stream:
                set_task_result(task_id=task_id, key="answer", value=state.get('answer'))
        
        
        def _build_prompt(self, state):
            max_context_chars =self.config.max_context_chars

            user_query = state.get('rewritten_query')

            item_names = state.get('item_names') or []

            retrieval_context = state.get('reranked_docs') or []
            formatted_context, usage_chars = self._format_retrieval_context(retrieval_context, max_context_chars)

            formatted_history = ""

            return ANSWER_PROMPT.format(
                context=formatted_context or "暂无检索上下文",
                history=formatted_history or "暂无历史上下文",
                item_names=','.join(item_names),
                question=user_query
            )
        
        def _format_retrieval_context(self, retrieval_context, max_context_chars):
            """
            格式化检索到的上下文
            【自己拼接一些元数据：供LLM学习，回答答案更准确】
            Args:
                retrieval_context: 检索到的上下文
                max_context_chars: 最大上下文的长度
            Returns:
                格式后的上下文
            """
            formatted_lines = []
            usage = 0
            for index, context in enumerate(retrieval_context, 1):
                content = context.get('content',"")

                if not content:
                    continue
                metadata_content = [f"[文档:{index}]"]

                for meta_field, template in [("chunk_id", "[chunk_id={}]"),
                                            ("title","[title={}]"),
                                            ("source","[source={}]"),
                                            ("url", "[url={}]")]:
                    field_value = str(context.get(meta_field,"")).strip()

                    if field_value:
                        metadata_content.append(template.format(field_value))
            
                doc_score = context.get('score')

                if doc_score is not None:
                    metadata_content.append(f"[score={float(doc_score):.6f}]")

                formatted_line = " ".join(metadata_content) + "\n" + content

                sep_chars = 2 if formatted_lines else 0

                total_length = sep_chars + len(formatted_line)

                if usage + total_length > max_context_chars:
                    break
                else:
                    formatted_lines.append(formatted_line)
                    usage += total_length
            return "\n\n".join(formatted_lines), max_context_chars - usage

