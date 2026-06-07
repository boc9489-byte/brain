import threading
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.answer_model_config import AnswerModelSettings
from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class AIClients(BaseClientManager):
    """AI 模型类客户端：OpenAI(VLM)"""

    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    _openai_llm_client: Optional[ChatOpenAI] = None
    _openai_llm_lock = threading.Lock()

    _answer_llm_clients = {}
    _answer_llm_lock = threading.Lock()

    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    # ── OpenAI / VLM ──

    @classmethod
    def get_bge_m3_openai(cls) :
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_openai)

    @classmethod
    def _create_bge_m3_openai(cls) :
        try:
            # bge_m3_ef = BGEM3EmbeddingFunction(
            #     model_name="/Users/bob/Documents/bge-m3", # Specify the model name
            #     device='mps', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            #     use_fp16=True # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            # )
            model_name = cls._require_env("BGE_M3_PATH")
            device = cls._require_env("BGE_DEVICE")
            use_fp16 = cls._require_env("BGE_FP16")

            use_fp16 = use_fp16.lower() in ("true","1")
            
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name, # Specify the model name
                device=device, # Specify the device to use, e.g., 'cpu' or 'cuda:0'
                use_fp16=use_fp16 # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
            )
            return bge_m3_ef

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"bge_m3 客户端创建失败: {e}")
            raise ConnectionError(f"bge_m3 连接失败: {e}") from e
    # ── OpenAI / VLM ──

    @classmethod
    def get_openai(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_openai)

    @classmethod
    def _create_openai(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e
    
        # ── ChatOpenAi / LLM ──

    @classmethod
    def get_llm_client(cls, response_format: bool =True) -> ChatOpenAI:
        return cls._get_or_create("_openai_llm_client", cls._openai_llm_lock, 
                                # cls._create_llm_openai(response_format) 立即执行
                                lambda : cls._create_llm_openai(response_format))

    @classmethod
    def get_answer_llm_client(cls, response_format: bool = False) -> ChatOpenAI:
        """获取业务回答模型客户端。

        阶段五把回答模型从通用 `OPENAI_API_BASE + LLM_DEFAULT_MODEL`
        拆成独立的 `ANSWER_*` 配置，便于 Base/SFT 灰度和回滚。
        """
        settings = AnswerModelSettings.from_env()
        issues = settings.issues()
        if issues:
            raise ConnectionError("Answer model config invalid: " + "; ".join(issues))

        cache_key = settings.cache_key(response_format)
        client = cls._answer_llm_clients.get(cache_key)
        if client is not None:
            return client

        with cls._answer_llm_lock:
            client = cls._answer_llm_clients.get(cache_key)
            if client is not None:
                return client
            client = cls._create_answer_llm_openai(settings, response_format)
            cls._answer_llm_clients[cache_key] = client
            return client

    @classmethod
    def _create_llm_openai(cls, response_format) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env("LLM_DEFAULT_MODEL")

            model_kwargs = {}
            if response_format:
                model_kwargs["response_format"]={"type": "json_object"}

            client = ChatOpenAI(
                model_name = model_name,
                openai_api_key=api_key, 
                openai_api_base=base_url,
                temperature=0,
                # ```json
                # {
                # "joke": "有一天，小明去面试，面试官问他：‘你有什么特长？’ 小明想了想，认真地说：‘我会预测未来。’ 面试官笑了笑：‘那你预测一下，你什么时候能被录用？’ 小明淡定地回答：‘这个嘛……我预测我不会被录用。’ 面试官一愣，随后笑着说：‘你被录用了！’ 小明叹了口气：‘唉，看来我的预测不准了。’"
                # }
                # ```
                # {
                model_kwargs=model_kwargs
                # model_kwargs={"response_format": {"type": "json_object"}}
                #     "joke": "有一天，小明去面试，面试官问他：‘你有什么特长？’ 小明想了想说：‘我会预测未来。’ 面试官笑了笑：‘那你预测一下，你什么时候能被录用？’ 小明淡定地说：‘这个嘛……我预测我不会被录用。’ 面试官一愣，随后笑着说：‘恭喜你，你被录用了！’ 小明叹了口气：‘唉，我的预测又不准了。’"
                # }

            )
            logger.info(f"OpenAI LLM 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    @classmethod
    def _create_answer_llm_openai(cls, settings: AnswerModelSettings, response_format: bool) -> ChatOpenAI:
        """创建业务回答模型客户端。

        vLLM LoRA 接入时，`model_name` 应等于 `--lora-modules` 中声明的 alias。
        例如：`--lora-modules kb-sft=/path/to/adapter` 对应 `ANSWER_SFT_MODEL=kb-sft`。
        """
        try:
            model_kwargs = {}
            if response_format:
                model_kwargs["response_format"] = {"type": "json_object"}

            client = ChatOpenAI(
                model_name=settings.model_name,
                openai_api_key=settings.api_key,
                openai_api_base=settings.base_url,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                request_timeout=settings.timeout_sec,
                model_kwargs=model_kwargs,
            )
            logger.info(
                "Answer LLM 客户端初始化成功 "
                f"(provider={settings.provider}, base_url={settings.base_url}, model={settings.model_name})"
            )
            return client
        except Exception as e:
            logger.error(f"Answer LLM 客户端创建失败: {e}")
            raise ConnectionError(f"Answer LLM 连接失败: {e}") from e

if __name__ == "__main__":

    llm_client = AIClients.get_llm_client()

    ll_response = llm_client.invoke("请给我讲一个笑话，输出格式是json")

    json = ll_response.content


    print(json)
