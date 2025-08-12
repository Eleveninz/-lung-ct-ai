# AI Lung Nodule Detection

## 项目简介
本项目旨在基于公开的肺部 CT 数据集（LIDC-IDRI），构建一个用于肺结节检测与分类的深度学习模型。  
目标是利用卷积神经网络（CNN）及深度学习技术，实现对医学影像的自动化分析，辅助医生进行早期肺癌筛查。

## 项目目标
- 数据加载与预处理（CT 扫描图像处理、标注解析）
- 模型搭建（CNN / 3D CNN / 迁移学习）
- 模型训练与评估（准确率、召回率、F1-score）
- 部署为可交互的 Web / 微信小程序

## 技术栈
- Python 3.x
- PyTorch / TensorFlow（待定）
- CUDA / cuDNN（GPU 加速）
- NumPy, Pandas, OpenCV
- Matplotlib / Seaborn

## 数据集
- **LIDC-IDRI**: 美国癌症影像档案（TCIA）提供的肺部 CT 扫描公开数据集
- 来源论文：[The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI)](https://doi.org/10.1118/1.3528204)

## 当前进度
- [x] 项目初始化
- [x] 数据集下载与解析
- [x] 数据预处理
- [x] 模型构建
- [x] 模型训练
- [ ] 部署与发布

## 项目结构（规划）

## 作者
- 大连医科大学 生物医学工程专业 大三学生  
- 对 AI、医疗影像处理、计算机视觉充满兴趣  
- 目标：进入科技前沿领域（AI、机器人、医疗器械等）

---

*备注：本项目为个人学习与探索之用，不用于商业医疗诊断。*
AI-Lung-Nodule-Detection/
│── README.md
│── requirements.txt
│── main.py
│── data/ # 数据集（本地存储，不上传到GitHub）
│── src/ # 代码目录
│ ├── preprocessing.py
│ ├── model.py
│ └── train.py
│── results/ # 模型输出与可视化结果
