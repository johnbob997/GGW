import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.stats import linregress

import numpy as np
import rasterio

# 读取第一个栅格数据
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\CROP\slope.tif') as src_A:
    A = src_A.read(1)
    transform_A = src_A.transform  # 获取地理参考信息
    rows, cols = A.shape

# 读取第一个显著区域图像
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\CROP\GGW_p_0.05.tif') as src:
    A_0 = src.read(1)

# 读取第二个栅格数据
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\TREE_0.05\slope.tif') as src_B:
    B = src_B.read(1)

# 读取第二个显著区域图像
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\TREE_0.05\GGW_p_0.05.tif') as src:
    B_0 = src.read(1)


with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\3_11\lpd_10000m\lpd_GGW_10000m.tif') as src:
    data = src.read(1)  # 读取第一个波段数据
    transform = src.transform  # 获取地理参考信息
    rows, cols = data.shape  # 获取栅格的行列数
    crs = src.crs  # 获取坐标参考系信息

# 创建初始化结果矩阵
# 这里我们用5作为初始化值，表示默认区域
result = np.full((rows, cols), 6)

# 标记存在值的区域为1
result[data != 0] = 5  # 将所有非零区域赋值为1
# 将A_0和B_0转换为布尔矩阵，非零值为True
presence_mask = (A_0 == 1) | (B_0 == 1)

# 将布尔矩阵转化为数值矩阵（1表示存在值，0表示不存在值）
p_pass = presence_mask.astype(np.uint8)
# with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\p_result.tif', 'w', driver='GTiff',
#                    height=rows, width=cols, count=1, dtype=result.dtype, crs=src_A.crs, transform=transform_A) as dst:
#     dst.write(p_pass, 1)
# 按条件划分四类区域
condition_A_increase_B_decrease = (p_pass > 0) & (A > 0) & (B < 0)
condition_A_decrease_B_increase = (p_pass > 0) & (A < 0) & (B > 0)
condition_A_increase_B_increase = (p_pass > 0) & (A > 0) & (B > 0)
condition_A_decrease_B_decrease = (p_pass > 0) & (A < 0) & (B < 0)

# 赋值
result[condition_A_increase_B_decrease] = 1
result[condition_A_decrease_B_increase] = 2
result[condition_A_increase_B_increase] = 3
result[condition_A_decrease_B_decrease] = 4

# 输出结果或保存为新栅格文件
# with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\result.tif', 'w', driver='GTiff',
#                    height=rows, width=cols, count=1, dtype=result.dtype, crs=src_A.crs, transform=transform_A) as dst:
#     dst.write(result, 1)
type_values = [1, 2, 3, 4]  # 五种类型的像元值
type_names = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
# 打开 TIF 文件
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\result.tif') as src:
    TIF_data = src.read(1)  # 读取第一波段数据
    TIF_data[TIF_data < 0] = 0  # 将无效值设为0
    left, bottom, right, top = src.bounds  # 获取地理边界
    num_pixels = TIF_data.shape[0]  # 纬度方向像元数
    latitudes = np.linspace(bottom, top, num_pixels)[::-1]  # 纬度数组（从北到南）
# 找到最接近 13.5 的纬度索引
split_index = np.abs(latitudes - 13.5).argmin()
print(np.linspace(bottom, top, num_pixels)[::-1])
# 初始化统计结果
north_counts = {t: 0 for t in type_values}
south_counts = {t: 0 for t in type_values}
print('split_index:',split_index)
# 统计像元数量
north_data = TIF_data[:split_index, :].flatten()
south_data = TIF_data[split_index:, :].flatten()

for t in type_values:
    north_counts[t] = np.sum(north_data == t)
    south_counts[t] = np.sum(south_data == t)

# 计算总像元数
north_total = np.sum(list(north_counts.values()))
south_total = np.sum(list(south_counts.values()))

# 计算占比
north_percentages = {type_names[t-1]: (count / north_total * 100) if north_total > 0 else 0
                     for t, count in north_counts.items()}
south_percentages = {type_names[t-1]: (count / south_total * 100) if south_total > 0 else 0
                     for t, count in south_counts.items()}

# 输出结果
print("13.5°N 以北类型占比：")
for t, perc in north_percentages.items():
    print(f"{t}: {perc:.2f}%")

print("\n13.5°N 以南类型占比：")
for t, perc in south_percentages.items():
    print(f"{t}: {perc:.2f}%")


