import matplotlib.pyplot as plt
import rasterio
import numpy as np
import pandas as pd
import seaborn as sns
import rasterio
import matplotlib.font_manager as fm
from scipy.stats import linregress
from matplotlib.font_manager import FontProperties
TIF = r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\RI_NEW\\GGW_0.1.tif'
TIF = rasterio.open(TIF)
band = TIF.read(1)
TIF_bounds = TIF.bounds  # 影像四至
print(TIF_bounds)
left,bottom,right,top = TIF.bounds
TIF_rows, TIF_cols = TIF.shape  # 影像行列号
print(TIF.shape)
TIF_bands = TIF.count  # 影像波段数
TIF_indexes = TIF.indexes  # 影像波段
TIF_crs = TIF.crs  # 影像坐标系
TIF_transform = TIF.transform  # 影像仿射矩阵
TIF_arr = TIF.read()
TIF_arr = np.nan_to_num(TIF_arr)
stats = []
for band in TIF_arr:
    stats.append({
        'min': band.min(),
        'mean': band.mean(),
        'median': np.median(band),
        'max': band.max()})
print(stats)
#path = r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\RI_NEW\\'
path = r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\3_11\3_11\\'

# TIF_pass = path + 'GGW_pass.tif'
# TIF_pass = rasterio.open(TIF_pass)
# TIF_pass = TIF_pass.read(1)
# TIF_all_p = np.nan_to_num(TIF_pass)

TIF_precipitation = path + 'GGW_precipitation.tif'
TIF_precipitation = rasterio.open(TIF_precipitation)
TIF_precipitation = TIF_precipitation.read(1)
TIF_precipitation = np.nan_to_num(TIF_precipitation)
TIF_precipitation = np.where(TIF_precipitation<0,0,TIF_precipitation)
#TIF_PRE = np.where(TIF_all_p==1,TIF_precipitation,0)

# TIF_temperature = path +'\\temperature_'+con+'.tif'
TIF_temperature = path + 'GGW_temperature.tif'
TIF_temperature = rasterio.open(TIF_temperature)
TIF_temperature = TIF_temperature.read(1)
TIF_temperature = np.nan_to_num(TIF_temperature)
TIF_temperature = np.where(TIF_temperature < 0, 0, TIF_temperature)
#TIF_TEM = np.where(TIF_all_p==1,TIF_temperature,0)

# TIF_radiation = path +'\\radiation_'+con+'.tif'
TIF_radiation = path + 'GGW_radiation.tif'
TIF_radiation = rasterio.open(TIF_radiation)
TIF_radiation = TIF_radiation.read(1)
TIF_radiation = np.nan_to_num(TIF_radiation)
TIF_radiation = np.where(TIF_radiation < 0, 0, TIF_radiation)
#TIF_RAD = np.where(TIF_all_p==1,TIF_radiation,0)

# TIF_crop = path +'\\crop_'+con+'.tif'
TIF_crop = path + 'GGW_crop.tif'
#TIF_crop = path + 'TN_crop.tif'
#TIF_crop = path + 'TN_crop'+con+'.tif'
TIF_crop = rasterio.open(TIF_crop)
TIF_crop = TIF_crop.read(1)
TIF_crop = np.nan_to_num(TIF_crop)
TIF_crop = np.where(TIF_crop < 0, 0, TIF_crop)
#TIF_CROP = np.where(TIF_all_p==1,TIF_crop,0)
count = sum(sum(np.where(TIF_crop>0,1,0)))
print(count)

# TIF_tree = path +'\\tree_'+con+'.tif'
TIF_tree = path + 'GGW_tree.tif'
#TIF_tree = path + 'TN_tree.tif'
#TIF_tree = path + 'TN_tree'+con+'.tif'
TIF_tree = rasterio.open(TIF_tree)
TIF_tree = TIF_tree.read(1)
TIF_tree = np.nan_to_num(TIF_tree)
TIF_tree = np.where(TIF_tree < 0, 0, TIF_tree)
#TIF_TREE = np.where(TIF_all_p==1,TIF_tree,0)
count = sum(sum(np.where(TIF_tree>0,1,0)))
print(count)

TIF_N = path + 'GGW_N.tif'
TIF_N = rasterio.open(TIF_N)
TIF_N = TIF_N.read(1)
TIF_N = np.nan_to_num(TIF_N)
TIF_N = np.where(TIF_N < 0, 0, TIF_N)
# TIF_CO2_p = np.where(TIF_all_p==1,TIF_CO2,0)

# TIF_CO2 = path +'\\CO2_'+con+'.tif'
TIF_CO2 = path + 'GGW_CO2.tif'
TIF_CO2 = rasterio.open(TIF_CO2)
TIF_CO2 = TIF_CO2.read(1)
TIF_CO2 = np.nan_to_num(TIF_CO2)
TIF_CO2 = np.where(TIF_CO2 < 0, 0, TIF_CO2)
# TIF_CO2_p = np.where(TIF_all_p==1,TIF_CO2,0)



TIF_all = TIF_precipitation + TIF_temperature + TIF_radiation + TIF_crop + TIF_tree + TIF_N + TIF_CO2
all = np.where(TIF_all > 0,1,0)
# print(con)
print('/precipitation:',round(100*TIF_precipitation.sum()/all.sum(),2),'/temperature:',round(100*TIF_temperature.sum()/all.sum(),2),'/radiation:',round(100*TIF_radiation.sum()/all.sum(),2),'/crop:',round(100*TIF_crop.sum()/all.sum(),2),'/tree:',round(100*TIF_tree.sum()/all.sum(),2),'/NDE:',round(100*TIF_N.sum()/all.sum(),2),'/CO2:',round(100*TIF_CO2.sum()/all.sum(),2))
#print('/precipitation:',round(100*TIF_PRE.sum()/TIF_all_p.sum(),2),'/temperature:',round(100*TIF_TEM.sum()/TIF_all_p.sum(),2),'/radiation:',round(100*TIF_RAD.sum()/TIF_all_p.sum(),2),'/crop:',round(100*TIF_CROP.sum()/TIF_all_p.sum(),2),'/tree:',round(100*TIF_TREE.sum()/TIF_all_p.sum(),2),'/CO2:',round(100*TIF_CO2_p.sum()/TIF_all_p.sum(),2))
print('------------------------------------')



from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np


TIF_class = [TIF_crop, TIF_tree]
var_name = ['Cropland', 'Forestland']
color = ['#a3644d', '#416f4f']

# 全局字体设置
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24

# 创建绘图
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_title("RI changes with latitude", fontsize=24)
ax.set_xlabel('Latitude', fontsize=24)
ax.set_ylabel('RI Value (%)', fontsize=24,labelpad=20)#, rotation=270)
# ax.yaxis.set_tick_params(labelrotation=270)
# ax.xaxis.set_tick_params(labelrotation=270)
# 存储 Cropland 和 Forestland 的拟合参数
crop_params, forest_params = None, None

# 遍历变量并绘制
for j in range(len(TIF_class)):
    TIF_data = 100 * TIF_class[j]
    # 获取影像四至和纬度信息
    left, bottom, right, top = TIF.bounds
    num_pixels = TIF.shape[0]  # 获取影像行数
    latitudes = np.linspace(bottom, top, num_pixels)[::-1]

    # 针对 Cropland 和 Forestland 仅考虑 RI > 0 的像素
    nonzero_mask = TIF_data > 0
    row_means = np.where(nonzero_mask.any(axis=1), TIF_data.sum(axis=1) / nonzero_mask.sum(axis=1), np.nan)

    # 展平数据用于拟合
    nonzero_indices = np.nonzero(TIF_data.flatten() > 0)
    values = TIF_data.flatten()[nonzero_indices]
    latitudes_expanded = np.repeat(latitudes, TIF_data.shape[1])[nonzero_indices]

    # 计算线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(latitudes_expanded, values)
    print(f"变量: {var_name[j]}，R²: {r_value ** 2:.4f}，p值: {p_value:.4f}")

    # 存储 Cropland 和 Forestland 的拟合参数
    if var_name[j] == 'Cropland':
        crop_params = (slope, intercept)
    elif var_name[j] == 'Forestland':
        forest_params = (slope, intercept)

    # 绘制线性拟合线
    ax.plot(latitudes_expanded, slope * latitudes_expanded + intercept, color=color[j], linestyle='-', linewidth=4,
            label=f'{var_name[j]}: y = {slope:.2f}x + {intercept:.2f}')

    # 绘制每行的 RI 均值折线
    ax.plot(latitudes, row_means, color=color[j], linestyle='--', linewidth=2, label=f'{var_name[j]} Mean RI')

# 计算并标记 Cropland 和 Forestland 拟合线的相交点
if crop_params and forest_params:
    # 交点纬度计算
    crop_slope, crop_intercept = crop_params
    forest_slope, forest_intercept = forest_params
    if crop_slope != forest_slope:  # 确保两条直线不平行
        intersection_lat = (forest_intercept - crop_intercept) / (crop_slope - forest_slope)
        intersection_ri = crop_slope * intersection_lat + crop_intercept

        # 在图中标记交点
        ax.plot(intersection_lat, intersection_ri, 'ro', markersize=10, label='Intersection')
        print(f"Cropland 和 Forestland 的交点: 纬度 = {intersection_lat:.2f}, RI = {intersection_ri:.2f}")

# 图例
# ax.legend(loc='upper left', fontsize=18)
plt.tight_layout()

# 保存或显示
#plt.savefig('linear_fits_with_mean_lines_all.jpg', dpi=600)
plt.show()



#-------------------------------------------------------------------

TIF_class = [TIF_precipitation, TIF_temperature, TIF_radiation, TIF_crop, TIF_tree, TIF_N, TIF_CO2]
var_name = ['Precipitation', 'Temperature', 'Radiation', 'Cropland', 'Forestland', 'NDE', 'CO$_{2}$']
color = ['#4e62ab', '#e0b6e6', '#c8c066', '#a3644d', '#416f4f', '#ffab1a', '#87d0a5']

# 全局字体设置
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 24

# 创建绘图
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_title("RI changes with latitude", fontsize=24)
ax.set_xlabel('Latitude', fontsize=24)
ax.set_ylabel('RI Value (%)', fontsize=24,labelpad=20)#, rotation=270)
# ax.yaxis.set_tick_params(labelrotation=270)
# ax.xaxis.set_tick_params(labelrotation=270)
# 存储 Cropland 和 Forestland 的拟合参数
crop_params, forest_params = None, None

# 遍历变量并绘制
for j in range(len(TIF_class)):
    TIF_data = 100 * TIF_class[j]
    # 获取影像四至和纬度信息
    left, bottom, right, top = TIF.bounds
    num_pixels = TIF.shape[0]  # 获取影像行数
    latitudes = np.linspace(bottom, top, num_pixels)[::-1]

    # 针对 Cropland 和 Forestland 仅考虑 RI > 0 的像素
    nonzero_mask = TIF_data > 0
    row_means = np.where(nonzero_mask.any(axis=1), TIF_data.sum(axis=1) / nonzero_mask.sum(axis=1), np.nan)

    # 展平数据用于拟合
    nonzero_indices = np.nonzero(TIF_data.flatten() > 0)
    values = TIF_data.flatten()[nonzero_indices]
    latitudes_expanded = np.repeat(latitudes, TIF_data.shape[1])[nonzero_indices]

    # 计算线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(latitudes_expanded, values)
    print(f"变量: {var_name[j]}，R²: {r_value ** 2:.4f}，p值: {p_value:.4f}")

    # 存储 Cropland 和 Forestland 的拟合参数
    if var_name[j] == 'Cropland':
        crop_params = (slope, intercept)
    elif var_name[j] == 'Forestland':
        forest_params = (slope, intercept)

    # 绘制线性拟合线
    ax.plot(latitudes_expanded, slope * latitudes_expanded + intercept, color=color[j], linestyle='-', linewidth=4,
            label=f'{var_name[j]}: y = {slope:.2f}x + {intercept:.2f}')

    # 绘制每行的 RI 均值折线
    ax.plot(latitudes, row_means, color=color[j], linestyle='--', linewidth=2, label=f'{var_name[j]} Mean RI')

# 计算并标记 Cropland 和 Forestland 拟合线的相交点
if crop_params and forest_params:
    # 交点纬度计算
    crop_slope, crop_intercept = crop_params
    forest_slope, forest_intercept = forest_params
    if crop_slope != forest_slope:  # 确保两条直线不平行
        intersection_lat = (forest_intercept - crop_intercept) / (crop_slope - forest_slope)
        intersection_ri = crop_slope * intersection_lat + crop_intercept

        # 在图中标记交点
        ax.plot(intersection_lat, intersection_ri, 'ro', markersize=10, label='Intersection')
        print(f"Cropland 和 Forestland 的交点: 纬度 = {intersection_lat:.2f}, RI = {intersection_ri:.2f}")

# 图例
# ax.legend(loc='upper left', fontsize=18)
plt.tight_layout()

# 保存或显示
#plt.savefig('linear_fits_with_mean_lines_all.jpg', dpi=600)
plt.show()
