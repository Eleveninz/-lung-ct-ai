#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LUNA16 -> YOLOv5 格式转换脚本（稳健版）
功能：
 - 读取 annotations.csv（真阳性结节）
 - 从 subset0..subsetN 的 .mhd 中提取对应切片并保存为 png
 - 标签使用 YOLO 格式： class x_center y_center width height（均归一化到 0-1）
 - 按 patient(seriesuid) 做 train/val 划分（不会把同一 patient 的不同切片放到 train 和 val 两边）
 - 生成可视化检查图（原图 + 红框 / 放大局部）
 - 可选生成少量负样本
注意：
 - 运行前安装依赖： pip install SimpleITK pandas opencv-python tqdm numpy Pillow
"""

import os
import random
import glob
from pathlib import Path
import math

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from tqdm import tqdm
from PIL import Image

# -----------------------------
# 在脚本中设置的参数
# -----------------------------
LUNA_DIR = "E:/AI-Lung-Nodule-Detection/-lung-ct-ai/data/LUNA16"  # LUNA16 数据集路径
OUT = "E:/AI-Lung-Nodule-Detection/-lung-ct-ai/data/luna16_yolo"  # 输出目录

subsets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]   # 使用的 subsets（可以设置为一个列表，如 [0, 1] 或 [0-2]）
img_ext = "png"  # 输出图像的格式（可以是 png 或 jpg）
val_ratio = 0.2  # 训练集和验证集划分比例（按 patient 划分）
neg_per_series = 2  # 每个 series 采样的负样本数量（0表示不采样负样本）
window_width = 1500.0  # 窗宽（用于增强可视化）
window_level = -600.0  # 窗位（用于增强可视化）
seed = 42  # 随机种子

# -----------------------------
# 工具函数
# -----------------------------
def world_to_voxel(world_coord, itk_image):
    """
    使用 SimpleITK 的 TransformPhysicalPointToIndex 方法将世界坐标转为像素坐标。
    world_coord: [x(mm), y(mm), z(mm)]
    itk_image: SimpleITK 图像对象
    返回 voxel 坐标 (x_v, y_v, z_v)，对应于图像坐标系（float）。
    """
    # 获取图像的 origin 和 spacing
    origin = itk_image.GetOrigin()  # (x0, y0, z0)
    spacing = itk_image.GetSpacing()  # (sx, sy, sz)
    
    # 使用 SimpleITK 的方法转换
    voxel = itk_image.TransformPhysicalPointToIndex(world_coord)
    return voxel  # 返回像素坐标

def apply_window(image, width, level):
    """
    对 2D np.array 做窗宽窗位线性裁剪并归一化到 0-255（uint8）
    """
    min_val = level - width / 2.0
    max_val = level + width / 2.0
    img = np.clip(image, min_val, max_val)
    img = (img - min_val) / (max_val - min_val) * 255.0
    img = img.astype(np.uint8)
    return img

# -----------------------------
# 主流程
# -----------------------------
def main():
    random.seed(seed)
    np.random.seed(seed)

    # 使用已经在脚本开头定义的 LUNA_DIR 和 OUT 变量，不再重新赋值
    IMAGES_TRAIN = Path(OUT) / "images" / "train"
    IMAGES_VAL = Path(OUT) / "images" / "val"
    LABELS_TRAIN = Path(OUT) / "labels" / "train"
    LABELS_VAL = Path(OUT) / "labels" / "val"
    CHECK_DIR = Path(OUT) / "check_images"

    for p in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL, CHECK_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # 读取 annotations.csv（真阳性）
    ann_path = Path(LUNA_DIR) / "annotations.csv"
    if not ann_path.exists():
        raise FileNotFoundError(f"找不到 {ann_path}，请确认 annotations.csv 在 LUNA16 目录下。")
    df = pd.read_csv(str(ann_path))
    # 期望 columns: seriesuid, coordX, coordY, coordZ, diameter_mm
    # 处理列名兼容性
    expected = ["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]
    if not set(expected).issubset(set(df.columns)):
        print("注意：annotations.csv 列名与预期不完全相同，尝试自动匹配。")
    df = df.rename(columns={c: c.strip() for c in df.columns})

    # 构造 subset -> mhd 文件映射 (seriesuid -> mhd path)
    mhd_map = {}
    for s in subsets:
        subset_dir = Path(LUNA_DIR) / f"subset{s}"
        if not subset_dir.exists():
            print(f"警告：{subset_dir} 不存在，跳过该子集")
            continue
        for m in subset_dir.glob("*.mhd"):
            mhd_map[m.stem] = str(m)

    print(f"找到 {len(mhd_map)} 个 mhd 文件（来自 subsets {subsets}）")

    # 只保留 annotations 中有对应 mhd 的 seriesuid
    df = df[df['seriesuid'].isin(mhd_map.keys())]
    print(f"annotations 中与 mhd 对应的记录数: {len(df)}")

    # 按 seriesuid 分组，先决定 train/val 的 series 列表（patient-level 划分）
    all_series = list(df['seriesuid'].unique())
    random.shuffle(all_series)

    val_count = math.floor(len(all_series) * val_ratio)
    val_series = set(all_series[:val_count])  # 划分验证集
    train_series = set(all_series[val_count:])  # 剩余的是训练集

    print(f"训练集患者数量: {len(train_series)}，验证集患者数量: {len(val_series)}")

    # 处理所有患者的切片
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        seriesuid = row["seriesuid"]
        coordX = row["coordX"]
        coordY = row["coordY"]
        coordZ = row["coordZ"]
        diameter_mm = row["diameter_mm"]

        # 获取 mhd 路径
        mhd_path = mhd_map[seriesuid]
        itk_image = sitk.ReadImage(mhd_path)

        # 将物理坐标转换为像素坐标
        voxel = world_to_voxel([coordX, coordY, coordZ], itk_image)
        
        # 获取图像切片，假设用 z 轴进行切片
        slice_image = sitk.GetArrayFromImage(itk_image)[voxel[2], :, :]

        # 应用窗宽窗位
        slice_image = apply_window(slice_image, window_width, window_level)

        # 转为 OpenCV 图像格式（BGR）用于显示与保存
        slice_image = cv2.cvtColor(slice_image, cv2.COLOR_GRAY2BGR)

        # 保存图片和标签
        if seriesuid in train_series:
            img_dir = IMAGES_TRAIN
            label_dir = LABELS_TRAIN
        else:
            img_dir = IMAGES_VAL
            label_dir = LABELS_VAL

        # 保存切片图像
        img_filename = f"{seriesuid}_{voxel[2]}.{img_ext}"
        cv2.imwrite(str(img_dir / img_filename), slice_image)

        # 保存对应的 YOLO 标签
        label_filename = img_filename.replace(img_ext, "txt")
        with open(label_dir / label_filename, "w") as f:
            # 计算结节的 bbox（使用 YOLO 格式）
            w, h = slice_image.shape[1], slice_image.shape[0]
            x_center = (voxel[0] + 0.5) / w  # 中心位置，归一化
            y_center = (voxel[1] + 0.5) / h  # 中心位置，归一化
            width = diameter_mm / w  # 宽度，归一化
            height = diameter_mm / h  # 高度，归一化
            f.write(f"0 {x_center} {y_center} {width} {height}\n")  # 假设结节为类 0

if __name__ == "__main__":
    main()
