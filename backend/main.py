import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from yolov8_infer import run_detection
from database import add_history, get_all_history
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前端地址，开发时可设为 ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.post("/api/detect/")
async def detect(file: UploadFile = File(...)):
    filename = file.filename
    suffix = os.path.splitext(filename)[-1].lower()
    if suffix not in ['.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov']:
        raise HTTPException(status_code=400, detail="不支持的文件格式")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 检测
    result_path, boxes = run_detection(file_path, RESULT_FOLDER)
    # 存入数据库
    add_history(filename, result_path, boxes)
    return {"result_image": result_path, "boxes": boxes}

@app.get("/api/result/")
def get_result(result_image: str):
    return FileResponse(result_image)

@app.get("/api/history/")
def history():
    return {"history": get_all_history()}