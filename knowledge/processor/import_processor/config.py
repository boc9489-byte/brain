from dataclasses import dataclass, field
from functools import lru_cache
from typing import FrozenSet
import os
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def require_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise ValueError(f"缺少环境变量: {key}")
    return value


def get_int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"环境变量 {key} 必须是整数，当前值: {value}")


def get_bool_env(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000
    img_content_length: int = 200
    min_content_length: int = 500
    overlap_sentences: int = 1
    item_name_chunk_k: int = 3
    item_name_chunk_size: int = 2500

    image_extensions: FrozenSet[str] = field(
        default_factory=lambda: frozenset(
            {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        )
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(default_factory=lambda: get_env("OPENAI_API_BASE"))
    openai_api_key: str = field(default_factory=lambda: require_env("OPENAI_API_KEY"))
    vl_model: str = field(default_factory=lambda: get_env("VL_MODEL"))
    item_model: str = field(default_factory=lambda: get_env("ITEM_MODEL"))
    default_model: str = field(default_factory=lambda: get_env("MODEL"))

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(default_factory=lambda: require_env("MILVUS_URL"))
    chunks_collection: str = field(default_factory=lambda: require_env("CHUNKS_COLLECTION"))
    item_name_collection: str = field(default_factory=lambda: require_env("ITEM_NAME_COLLECTION"))

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(default_factory=lambda: require_env("MINIO_ENDPOINT"))
    minio_access_key: str = field(default_factory=lambda: require_env("MINIO_ACCESS_KEY"))
    minio_secret_key: str = field(default_factory=lambda: require_env("MINIO_SECRET_KEY"))
    minio_bucket: str = field(default_factory=lambda: require_env("MINIO_BUCKET_NAME"))
    minio_secure: bool = field(default_factory=lambda: get_bool_env("MINIO_SECURE", False))

    # ==================== 向量配置 ====================
    embedding_dim: int = field(default_factory=lambda: get_int_env("EMBEDDING_DIM", 1024))
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15

    def __post_init__(self):
        if self.max_content_length <= 0:
            raise ValueError("max_content_length 必须大于 0")
        if self.min_content_length < 0:
            raise ValueError("min_content_length 不能小于 0")
        if self.min_content_length > self.max_content_length:
            raise ValueError("min_content_length 不能大于 max_content_length")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim 必须大于 0")

    def get_minio_base_url(self) -> str:
        protocol = "https://" if self.minio_secure else "http://"
        return f"{protocol}{self.minio_endpoint}"

# # ==================== 全局单例 ====================
# _config: Optional[ImportConfig] = None


# def get_config() -> ImportConfig:
#     """获取配置单例"""
#     global _config
#     if _config is None:
#         _config = ImportConfig.from_env()
#     return _config


@lru_cache(maxsize=1)
def get_config() -> ImportConfig:
    return ImportConfig()

# """
# 导入流程配置管理模块

# 集中管理所有配置项，支持环境变量覆盖
# """

# from dataclasses import dataclass, field
# from typing import Set, Optional
# import os
# from dotenv import load_dotenv

# load_dotenv()


# @dataclass
# class ImportConfig:
#     """导入流程配置"""

#     # ==================== 文档处理配置 ====================
#     max_content_length: int = 2000  # 切片最大长度
#     img_content_length: int = 200  # 图片上下文最大长度
#     min_content_length: int = 500  # 合并短内容的最小长度
#     overlap_sentences: int = 1  # 句子级切分时的重叠句数
#     item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
#     item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容长度

#     image_extensions: Set[str] = field(
#         default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
#     )

#     # ==================== LLM 配置 ====================
#     openai_api_base: str = field(
#         default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
#     )
#     openai_api_key: str = field(
#         default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
#     )
#     vl_model: str = field(
#         default_factory=lambda: os.getenv("VL_MODEL", "")
#     )
#     item_model: str = field(
#         default_factory=lambda: os.getenv("ITEM_MODEL", "")
#     )
#     default_model: str = field(
#         default_factory=lambda: os.getenv("MODEL", "")
#     )

#     # ==================== Milvus 配置 ====================
#     milvus_url: str = field(
#         default_factory=lambda: os.getenv("MILVUS_URL", "")
#     )
#     chunks_collection: str = field(
#         default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
#     )
#     item_name_collection: str = field(
#         default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
#     )
#     entity_name_collection: str = field(
#         default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
#     )


#     # ==================== MinIO 配置 ====================
#     minio_endpoint: str = field(
#         default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
#     )
#     minio_access_key: str = field(
#         default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
#     )
#     minio_secret_key: str = field(
#         default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
#     )
#     minio_bucket: str = field(
#         default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
#     )
#     minio_secure: bool = False

#     # ==================== 向量配置 ====================
#     embedding_dim: int = field(
#         default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
#     )
#     embedding_batch_size: int = 8

#     # ==================== 速率限制 ====================
#     requests_per_minute: int = 15  # 图片总结 API 速率限制

#     @classmethod
#     def from_env(cls) -> "ImportConfig":
#         """从环境变量加载配置"""
#         return cls()

#     # http://192.168.200.130:9000/
#     def get_minio_base_url(self):
#         base_protocol = "https://" if self.minio_secure else "http://"
#         return base_protocol + f"{self.minio_endpoint}"


# # ==================== 全局单例 ====================
# _config: Optional[ImportConfig] = None


# def get_config() -> ImportConfig:
#     """获取配置单例"""
#     global _config
#     if _config is None:
#         _config = ImportConfig.from_env()
#     return _config
