from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import torch
import os
import uuid
from PIL import Image
from torchvision import transforms
from flask_cors import CORS  # 新增：处理跨域

app = Flask(__name__)
CORS(app)  # 新增：启用跨域

# 加载模型
try:
    model = torch.hub.load(
        'E:/AI-Lung-Nodule-Detection/-lung-ct-ai/yolov5',  # 本地 YOLOv5 文件夹绝对路径
        'custom', 
        path=r'E:\AI-Lung-Nodule-Detection\-lung-ct-ai\yolov5\runs\train\luna16_model\weights\best.pt',
        source='local',  # 本地加载标识
        force_reload=False  # 无需强制重新下载
    )
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None  # 加载失败时标记为None，后续请求直接返回错误

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/detect', methods=['POST'])
def detect():
    if model is None:
        return jsonify({"error": "模型加载失败，请检查后端日志"}), 500

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "未收到图片"}), 400

    results = []

    for file in files:
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        try:
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
                    "leaf": "位置未知",  # 可后续扩展为智能判断
                    "diameter_mm": float(row['xmax'] - row['xmin'])  # 示例：像素差（需后续校准为实际毫米）
                })

            results.append({
                "image_url": f"http://localhost:5000/uploads/{filename}",  # 修正：正确的HTTP路径
                "bboxes": boxes
            })
        except Exception as e:
            print(f"处理图片 {filename} 失败: {e}")
            # 可选：记录错误后，跳过该图片或返回部分结果

    # 生成报告
    total_nodules = sum(len(r['bboxes']) for r in results)
    report = f"共检测到 {total_nodules} 个疑似肺结节，建议结合临床进一步复查。"

    return jsonify({"slices": results, "report": report})

@app.route('/uploads/<path:filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)