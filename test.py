from pathlib import Path
import shutil  # 单独导入shutil模块

img_dir = Path(r"E:\AI-Lung-Nodule-Detection\-lung-ct-ai\data\luna16_yolo\images\val")
label_dir = Path(r"E:\AI-Lung-Nodule-Detection\-lung-ct-ai\data\luna16_yolo\labels\val")
out = Path(r"E:\AI-Lung-Nodule-Detection\test_images_for_demo")
out.mkdir(parents=True, exist_ok=True)

for txt in label_dir.glob("*.txt"):
    img = img_dir / (txt.stem + ".png")
    if img.exists():
        shutil.copy(img, out / img.name)

print("copied labeled images to", out)