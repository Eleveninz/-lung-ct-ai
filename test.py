import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

# 这里换成你解压的位置
data_dir = r"D:\LUNA16\subset0"

# 找到一个 mhd 文件
for file in os.listdir(data_dir):
    if file.endswith(".mhd"):
        mhd_path = os.path.join(data_dir, file)
        break

# 读取 3D CT 数据
itk_img = sitk.ReadImage(mhd_path)
ct_array = sitk.GetArrayFromImage(itk_img)  # shape: [切片数, 高, 宽]

print("CT 体数据形状:", ct_array.shape)

# 可视化第 50 张切片
plt.imshow(ct_array[50], cmap='gray')
plt.title("Slice 50")
plt.show()
