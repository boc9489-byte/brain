
from datetime import datetime
from fileinput import filename
import os
import logging

# from time import time
import shutil
import time
from uuid import uuid4
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError

from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.task_util import add_node_duration, update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, add_running_task, add_done_task


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UploadService:
    """文件上传"""

    def get_base_dir(self) -> str:
        return os.path.join(get_local_base_dir(),datetime.now().strftime("%Y%m%d"))

    def run_import_graph(self, import_file_path:str, file_dir:str, task_id:str):
        """运行Langraph"""
            # import_graph()

        update_task_status(task_id, TASK_STATUS_PROCESSING)
        
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }
        try:
            for event in import_app.stream(graph_state):
                for key, value in event.items():
                    logger.info(f"当前正在运行的节点：->> {key}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"{task_id} 导入过程中出现异常， 原因： {str(e)}")
            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self, file:UploadFile):
        """文件上传"""
        # 文件目录
        task_id = uuid4().hex[:8] 
        add_running_task(task_id, "upload_file")
        # add_running_task(task_id, "upload_file")
        start_time = time.time()
        base_dir = self.get_base_dir()

        file_dir = os.path.join(base_dir, task_id)
        # 写入本地
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 写入minio
        self.save_upload_file_to_minio(import_file_path, filename)
        print(f"add_done_task {task_id = } upload_file")
        add_done_task(task_id, "upload_file")
        add_node_duration(task_id,"upload_file", time.time() - start_time)
        # add_done_task(task_id, "upload_file")x
        # add_node_duration(task_id, "upload_file", end_time - start_time)

        return import_file_path,file_dir, task_id



    def save_upload_file_to_minio(self, import_file_path:str, filename:str):
        try:
            minio_client = StorageClients.get_minio()
        except ConnectionError as e:
            logger.error(f"MinIO客户端获取失败，原因：{str(e)}")
            return 
        
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        object_name = f"origin_files/{datetime.now().strftime("%Y%m%d")}/{filename}"

        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)

        except Exception as e:
            logger.error(f"{filename}上传到MinIO失败，原因：{str(e)}")

    def save_upload_file_to_local(self, file:UploadFile, file_dir:str):
        """本地上传"""
        # 目录
        os.makedirs(file_dir, exist_ok=True)

        import_file_path = os.path.join(file_dir, file.filename)
        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.info(f"上传{file.filename}写入临时目录失败，原因：{str(e)}")
            raise FileProcessingError(message=f"上传{file.filename}写入临时目录失败，原因：{str(e)}")

        return import_file_path

