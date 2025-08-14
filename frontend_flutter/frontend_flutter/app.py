from flask import Flask, request, jsonify
from pathlib import Path
import torch
import os
import uuid
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

# 加载模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/new_model/weights/best.pt', force_reload=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/detect', methods=['POST'])
def detect():
    files = request.files.getlist('images')
    results = []

    for file in files:
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = Image.open(filepath).convert('RGB')
        res = model(img, size=416)
        preds = res.pandas().xyxy[0]

        boxes = []
        for _, row in preds.iterrows():
            boxes.append({
                "x1": int(row['xmin']),
                "y1": int(row['ymin']),
                "x2": int(row['xmax']),
                "y2": int(row['ymax']),
                "conf": float(row['confidence']),
                "leaf": "位置未知",  # 你可以改为智能判断
                "diameter_mm": float(row['xmax'] - row['xmin'])  # 示例
            })

        results.append({
            "image_url": f"http://localhost:5000/{filepath}",
            "bboxes": boxes
        })

    # 简单报告
    report = f"检测到 {sum(len(r['bboxes']) for r in results)} 个疑似结节，建议复查。"

    return jsonify({"slices": results, "report": report})

@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return app.send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
