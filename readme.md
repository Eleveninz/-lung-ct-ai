# AI Lung Nodule Detection 🫁🧠

基于 LUNA16 数据集的肺结节自动检测系统，支持上传 CT 切片图像，通过 YOLOv5 模型实现结节检测，生成可视化结果与结构化诊断报告。系统采用 **PyTorch 后端 + Flutter 跨平台前端** 架构，适用于医学影像 AI 项目展示、学习与二次开发。

---

## 🔧 技术栈

| 组件       | 技术选型                 | 核心作用                     |
|------------|--------------------------|------------------------------|
| 模型训练   | YOLOv5 (PyTorch)         | 肺结节目标检测模型训练与推理 |
| 数据处理   | SimpleITK, OpenCV        | DICOM格式解析、图像预处理    |
| 前端界面   | Flutter (Web/Android/iOS)| 跨平台交互界面（文件上传、结果展示） |
| 后端服务   | Flask REST API           | 衔接前端与模型，提供推理接口 |
| 报告生成   | OpenAI GPT API（可选）   | 生成医学风格诊断报告         |
| 推理加速   | CUDA（GPU）/CPU          | 本地/云端推理性能优化        |

---

## 🚀 快速启动

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Eleveninz/-lung-ct-ai.git
cd ./-lung-ct-ai  # 仓库名以"-"开头，需用"./"避免命令解析错误
```

### 2️⃣ 环境配置与依赖安装

#### 推荐使用 Conda 环境（避免版本冲突）
```bash
# 1. 创建并激活环境
conda create -n lung_ai python=3.8
conda activate lung_ai

# 2. 安装后端核心依赖
pip install -r requirements.txt

# 3. 安装 YOLOv5 额外依赖（训练/推理专用）
cd yolov5
pip install -r requirements.txt
cd ..  # 返回项目根目录
```

### 3️⃣ YOLOv5 模型训练（核心步骤）

#### 3.1 数据集准备
- **数据集来源**：LIDC-IDRI（美国癌症影像档案 TCIA 公开肺部 CT 数据集）
  - 论文参考：[The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)](https://doi.org/10.1118/1.3528204)
  - 下载地址：[TCIA 官网 LIDC-IDRI 页面](https://www.cancerimagingarchive.net/collections/lidc-idri/)
- **数据集存放**：下载后解压至 `data/raw/` 目录（目录结构见下文）

#### 3.2 数据预处理（格式转换与标注提取）
```bash
# 运行预处理脚本：将DICOM转为PNG，提取结节标注（适配YOLOv5格式）
python covert_luna16_to_yolo.py
  --input data/raw/ \  # 原始数据集路径自行设置
  --output data/processed/ \  # 预处理后输出路径自行设置
  --size 416  # 图像统一缩放尺寸（与训练输入尺寸一致）
```

#### 3.3 训练配置文件
- 数据集配置：`data/luna16.yaml`（已预设训练/验证集路径、类别数（仅"lung_nodule"1类））
- 模型配置：默认用 YOLOv5s（轻量级，适合快速部署），路径 `yolov5/models/yolov5s.yaml`

#### 3.4 启动训练
```bash
# 单GPU训练（基础命令）
python yolov5/train.py \
  --img 416 \  # 输入图像尺寸
  --batch 16 \  # 批次大小（根据GPU显存调整，显存不足可设8）
  --epochs 100 \  # 训练轮次（建议至少80轮保证精度）
  --data data/luna16.yaml \  # 数据集配置文件
  --cfg yolov5/models/yolov5s.yaml \  # 模型结构配置
  --weights '' \  # 从头训练（如需预训练权重，替换为'yolov5s.pt'）
  --name lung_nodule_train \  # 训练任务名称（日志/权重存放用）
  --cache  # 缓存图像到内存，加速训练
```

#### 3.5 训练监控
- 训练日志与权重：存于 `runs/train/lung_nodule_train/`（含 `best.pt` 最优权重、`results.csv` 指标日志）
- TensorBoard 可视化：
  ```bash
  tensorboard --logdir runs/train/  # 浏览器访问 localhost:6006 查看损失、mAP等曲线
  ```

### 4️⃣ 本地推理测试（验证模型效果）

```bash
python detect.py \
  --weights runs/train/lung_nodule_train/weights/best.pt \  # 训练好的最优权重
  --source test_images/ct_sample.png \  # 测试图片路径（支持单图/文件夹）
  --img 640 \  # 推理图像尺寸（与训练一致）
  --conf 0.5 \  # 置信度阈值（过滤低可信度检测结果）
  --save-txt \  # 可选：保存检测框坐标到txt文件
  --save-conf \  # 可选：保存检测结果置信度
  --project runs/detect \  # 推理结果保存路径
  --name lung_nodule_infer  # 推理任务名称
```

### 5️⃣ 启动前后端服务（完整系统体验）
```bash
# 1. 启动Flask后端（默认端口5000）
cd flask_server
python app.py

# 2. 启动Flutter前端（Web端示例）
cd ../frontend_flutter
flutter run -d chrome  # 需提前安装Flutter环境
```

---

## 🖼️ 系统核心功能

1. **CT 图像上传与解析**  
   - 支持格式：JPG/PNG（切片图）、DICOM（原始CT文件）
   - 前端实时预览，自动适配图像尺寸

2. **肺结节 AI 检测**  
   - 输出结果：结节边界框（x1,y1,x2,y2）、置信度（0-1）
   - 检测速度：单张图推理＜1秒（GPU加速）

3. **检测结果可视化**  
   - 叠加显示结节框，支持图像缩放/平移
   - 点击结节查看详细信息（位置、置信度）

4. **结构化诊断报告生成**  
   - 报告内容：结节数量、位置、尺寸估算、风险等级（基于LIDC标准）、临床建议
   - 功能：PDF下载、链接分享（可选对接GPT优化语言表述）

---

## 📂 项目目录结构

```
lung-ct-ai/
├── data/                 # 数据集相关
│   ├── raw/              # 原始LIDC-IDRI数据集（需自行下载）
│   ├── processed/        # 预处理后的图像+YOLO格式标注
│   ├── luna16.yaml       # YOLOv5训练数据集配置文件
│   └── preprocess.py     # 数据预处理脚本（DICOM转PNG、标注提取）
├── flask_server/         # 后端服务
│   ├── app.py            # Flask API主程序（提供上传、推理接口）
│   ├── inference.py      # 模型加载与推理核心逻辑
│   └── report.py         # 诊断报告生成模块
├── frontend_flutter/     # 跨平台前端
│   ├── lib/              # Dart源代码（界面、接口调用）
│   ├── assets/           # 静态资源（图标、示例图）
│   └── web/              # Web端构建产物（部署用）
├── yolov5/               # YOLOv5模型代码（含训练/推理）
│   ├── train.py          # 训练脚本
│   ├── detect.py         # 推理脚本
│   └── models/           # 模型结构配置文件
├── test_images/          # 测试用CT切片示例（可直接用于推理测试）
├── requirements.txt      # 后端核心依赖清单
└── README.md             # 项目说明文档（本文档）
```

---

## 📊 当前进度

- [x] 项目初始化（前后端框架搭建）
- [x] LIDC-IDRI 数据集下载与解析
- [x] 数据预处理（DICOM转码、标注格式转换）
- [x] YOLOv5 模型构建与训练（支持单/多GPU）
- [x] 本地推理功能验证
- [x] 前后端接口联调（上传-检测-展示）
- [ ] 云端部署（支持在线访问）
- [ ] 报告生成模块优化（对接更多临床标准）

---

## 🙋‍♂️ 作者信息

- 🎓 大连医科大学 生物医学工程专业
- 💡 研究方向：人工智能在医学影像处理中的应用
- 🚀 项目定位：AI 求职作品集（医学影像方向）
- 📫 代码仓库：[GitHub - Eleveninz/-lung-ct-ai](https://github.com/Eleveninz/-lung-ct-ai)
