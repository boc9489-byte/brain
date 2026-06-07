from typing import Any, Dict, List, Tuple

from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.sse_util import SSEEvent, push_sse_event
from knowledge.utils.task_util import set_task_result
from knowledge.prompt.query_prompy import ANSWER_PROMPT


class AnswerOutPutNode(BaseNode):
    """查询链路的最终回答节点。

    阶段五只替换回答模型客户端获取方式，不改变检索、rerank 和 prompt 拼接流程。
    默认走 base；当 `ANSWER_MODEL_PROVIDER=sft` 时，切到 vLLM LoRA alias。
    """

    name = "answer_output_node"

    def process(self, state: T) -> T:
        """生成或推送最终答案。"""
        is_stream = bool(state.get("is_stream"))
        task_id = state.get("task_id", "")

        if state.get("answer"):
            self._push_exist_answer(task_id, is_stream, state)
            is_streamed = False
        else:
            prompt = self._build_prompt(state)
            state["prompt"] = prompt
            self._generate_answer(prompt, task_id, state)
            is_streamed = is_stream

        if is_stream:
            if is_streamed:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
            else:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": state.get("answer")})

        return state

    def _generate_answer(self, prompt: str, task_id: str, state: Dict[str, Any]) -> None:
        """调用回答模型并写回 state。

        这里使用阶段五新增的 `get_answer_llm_client`，让回答链路能通过环境变量
        在 base 和 sft 之间切换；导入链路仍然使用原来的 LLM 客户端。
        """
        try:
            llm_client = AIClients.get_answer_llm_client(response_format=False)
        except ConnectionError as exc:
            self.logger.error(f"获取回答模型客户端失败，原因：{exc}")
            state["answer"] = "LLM暂无回答"
            if state.get("is_stream"):
                push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": state["answer"]})
            else:
                set_task_result(task_id=task_id, key="answer", value=state["answer"])
            return

        if state.get("is_stream"):
            state["answer"] = self._stream_llm(task_id, prompt, llm_client)
            return

        state["answer"] = self._invoke_llm(prompt, llm_client)
        set_task_result(task_id=task_id, key="answer", value=state["answer"])

    def _invoke_llm(self, prompt: str, llm_client: Any) -> str:
        """非流式生成答案。"""
        try:
            llm_res = llm_client.invoke(prompt)
            llm_content = getattr(llm_res, "content", "") or ""
            return llm_content or "LLM暂无回答"
        except Exception as exc:
            self.logger.error(f"LLM 非流式调用失败，原因：{exc}")
            return "LLM暂无回答"

    def _stream_llm(self, task_id: str, prompt: str, llm_client: Any) -> str:
        """流式生成答案，并把 delta 推送到 SSE。"""
        answer = ""
        try:
            for chunk in llm_client.stream(prompt):
                delta_text = getattr(chunk, "content", "") or ""
                if not delta_text:
                    continue
                push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": delta_text})
                answer += delta_text
        except Exception as exc:
            self.logger.error(f"LLM 流式调用失败，原因：{exc}")
            fallback = "LLM暂无回答"
            push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": fallback})
            return fallback
        if not answer:
            fallback = "LLM暂无回答"
            push_sse_event(task_id=task_id, event=SSEEvent.DELTA, data={"delta": fallback})
            return fallback
        return answer

    def _push_exist_answer(self, task_id: str, is_stream: bool, state: Dict[str, Any]) -> None:
        """已有答案时直接返回，避免重复调用模型。"""
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get("answer"))

    def _build_prompt(self, state: Dict[str, Any]) -> str:
        """把 rerank 后的文档拼成回答 prompt。"""
        max_context_chars = int(getattr(self.config, "max_context_chars", 6000))
        user_query = state.get("rewritten_query") or state.get("query") or ""
        item_names = state.get("item_names") or []
        retrieval_context = state.get("reranked_docs") or []
        formatted_context, _usage_chars = self._format_retrieval_context(retrieval_context, max_context_chars)

        return ANSWER_PROMPT.format(
            context=formatted_context or "暂无检索上下文",
            history="暂无历史上下文",
            item_names=",".join(item_names),
            question=user_query,
        )

    def _format_retrieval_context(
        self,
        retrieval_context: List[Dict[str, Any]],
        max_context_chars: int,
    ) -> Tuple[str, int]:
        """格式化检索上下文，并控制 prompt 长度。"""
        formatted_lines: List[str] = []
        usage = 0
        for index, context in enumerate(retrieval_context, 1):
            content = context.get("content", "")
            if not content:
                continue

            metadata = [f"[文档:{index}]"]
            for meta_field, template in [
                ("chunk_id", "[chunk_id={}]"),
                ("title", "[title={}]"),
                ("source", "[source={}]"),
                ("file_title", "[file_title={}]"),
                ("url", "[url={}]"),
            ]:
                field_value = str(context.get(meta_field, "")).strip()
                if field_value:
                    metadata.append(template.format(field_value))

            doc_score = context.get("score")
            if doc_score is not None:
                metadata.append(f"[score={float(doc_score):.6f}]")

            formatted_line = " ".join(metadata) + "\n" + content
            sep_chars = 2 if formatted_lines else 0
            total_length = sep_chars + len(formatted_line)
            if usage + total_length > max_context_chars:
                break

            formatted_lines.append(formatted_line)
            usage += total_length

        return "\n\n".join(formatted_lines), max_context_chars - usage
