import os
import uvicorn
from fastapi import Depends, FastAPI, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# from 

from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.upload_service import UploadService
from knowledge.core.deps import get_upload_file_service
from knowledge.utils.task_util import get_task_info

def create_app():
    app = FastAPI(description="掌柜智库导入的应用", version="v1.0")

    # 跨域 CORS
    #     def __init__(
    #     self,
    #     app: ASGIApp,
    #     allow_origins: Sequence[str] = (),
    #     allow_methods: Sequence[str] = ("GET",),
    #     allow_headers: Sequence[str] = (),
    #     allow_credentials: bool = False,
    #     allow_origin_regex: str | None = None,
    #     allow_private_network: bool = False,
    #     expose_headers: Sequence[str] = (),
    #     max_age: int = 600,
    # ) -> None:
    #     if "*" in allow_methods:
    #         allow_methods = ALL_METHODS

    app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_credentials = False,
        allow_methods= ["*"],
        allow_headers=["*"],
    )
    # 静态文件挂载
    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory = page_dir))


    # app.add_route(app)
    register_router(app)

    return app

def register_router(app:FastAPI):


    @app.get("/hello")
    def hello_world():
        return {"flag":"success"}

    @app.post("/upload",response_model = UploadResponse)
    def upload_endpoint(file: UploadFile,
                        backgroud_task: BackgroundTasks,
                        upload_service: UploadService = Depends(get_upload_file_service),
                        
                            ):  
        print("filename",file.filename)
        # upload_service = UploadService()
        import_file_path,file_dir, task_id = upload_service.process_upload_file(file)

        # upload_service.run_import_graph(import_file_path,file_dir, task_id)

        backgroud_task.add_task(upload_service.run_import_graph, import_file_path,file_dir, task_id)

        return UploadResponse(message=f"{file.filename}文件上传成功" , task_id=task_id)



    @app.get("/status/{task_id}")
    def get_task_status_endpoint(task_id:str):
        task_info = get_task_info(task_id)

        return  TaskStatusResponse(**task_info)

if __name__ =="__main__":
    uvicorn.run(app=create_app(),host="localhost",port=8000,log_level="info") 
