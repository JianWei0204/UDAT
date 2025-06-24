import os
from ultralytics import YOLO
import cv2

model = YOLO("yolov8/weights/yolov8n.pt")  # 请确保权重文件在此路径

def run_detection(img_path, result_folder):
    # 推理
    results = model(img_path)
    boxes = []
    for r in results:
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            boxes.append({'bbox': xyxy, 'confidence': conf, 'class': cls})
        img = r.plot()  # 带检测框的图片
        result_img_path = os.path.join(result_folder, os.path.basename(img_path))
        cv2.imwrite(result_img_path, img)
    return result_img_path, boxes