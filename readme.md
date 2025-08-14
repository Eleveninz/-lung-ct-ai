# AI Lung Nodule Detection 🫁🧠

基于 LUNA16 数据集的肺结节自动检测系统，支持上传 CT 切片图像，通过 YOLOv5 检测结节，并生成可视化结果与简要诊断报告。系统由 PyTorch 后端 + Flutter 跨平台前端组成，适用于医学影像 AI 项目展示与学习。

---

## 🔧 技术栈

| 组件       | 技术                     |
|------------|--------------------------|
| 模型训练   | YOLOv5 (PyTorch)         |
| 数据处理   | SimpleITK, OpenCV        |
| 前端界面   | Flutter (Web/Android/iOS)|
| 后端服务   | Flask REST API           |
| 报告生成   | OpenAI GPT API（可选）   |
| 推理平台   | 本地或云端（支持GPU）    |

---

## 🚀 快速启动

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Eleveninz/-lung-ct-ai.git
cd -lung-ct-ai

### 2️⃣ 安装依赖

# 建议使用 conda 环境
conda activate lung_ai
pip install -r requirements.txt

### 3️⃣ 模型训练（YOLOv5）
## 数据集
- **LIDC-IDRI**: 美国癌症影像档案（TCIA）提供的肺部 CT 扫描公开数据集
- 来源论文：[The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)](https://doi.org/10.1118/1.3528204)
### 4️⃣ 本地推理测试
python detect.py --weights runs/train/your_model/weights/best.pt --source test_images/
🖼️ 系统功能演示

上传 CT 切片（支持 JPG / PNG / DICOM）

YOLOv5 模型检测肺结节区域，返回置信度与框坐标

Flutter 页面显示检测图与坐标

一键生成医学风格报告（可下载 / 可分享）

lung-ct-ai/
├── data/                 # LUNA16 数据集及预处理
├── flask_server/         # 后端推理服务（Flask）
├── frontend_flutter/          # Flutter 前端界面
├── yolov5/               # YOLOv5 检测模型（含训练）
├── requirements.txt
└── README.md

## 当前进度
- [x] 项目初始化
- [x] 数据集下载与解析
- [x] 数据预处理
- [x] 模型构建
- [x] 模型训练
- [ ] 部署与发布

🙋‍♂️ 作者简介

🎓 大连医科大学 生物医学工程专业

💡 热爱人工智能 + 医学影像处理

🚀 本项目为 AI 求职作品集之一

📫 GitHub: Eleveninz

