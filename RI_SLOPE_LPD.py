import rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.polynomial import Polynomial
import statsmodels.api as sm


# 设置Times New Roman字体
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体为Times New Roman
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 16  # 设置全局字体大小为24号

# 读取tif文件
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\3_11\lpd_10000m\lpd_GGW_10000m.tif') as src_A:
    A = src_A.read(1)  # 读取第一波段
    transform_A = src_A.transform  # 获取地理参考信息
    height, width = A.shape

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\CROP\slope.tif') as src:
    B = src.read(1)

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\CROP\GGW_p_0.05.tif') as src:
    C = src.read(1)
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\3_11\GGW\GGW_crop.tif') as src:  # 读取图像D
    D = src.read(1)

# # 读取两个图像C1和C2
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\CROP\GGW_p_0.05.tif') as src1:
    C1 = src1.read(1)

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\TREE_0.05\GGW_p_0.05.tif') as src2:
    C2 = src2.read(1)

# 创建至少存在值的mask
mask_agg = (~np.isnan(C1)) | (~np.isnan(C2))

C = mask_agg

# 应用掩膜过滤
A_masked = np.where(C == 1, A, np.nan)
B_masked = np.where(C == 1, B, np.nan)
D_padded = np.pad(D, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
D_masked = np.where(C == 1, D_padded, np.nan)

# 获取图像的行和列数
rows, cols = A.shape

# 获取每个像素的地理坐标（纬度和经度）
x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
latitudes, longitudes = rasterio.transform.xy(transform_A, y_coords, x_coords)
latitudes = np.array(latitudes)

A_types = [1, 5]
colors = {1: 'red', 5: 'green'}
B_bins = np.linspace(np.nanmin(B_masked), np.nanmax(B_masked), 100)
# Define the label mapping
label_mapping = {
    1: 'Declining',
    2: 'Early signs of decline',
    3: 'Stable but stressed',
    4: 'Stable and not stressed',
    5: 'Increasing'
}

# 定义加权移动平均方法
def weighted_moving_average(data, weights):
    return np.convolve(data, weights / weights.sum(), mode='same')

weights = np.array([1, 2, 3, 2, 1])  # 权重设置

# 创建一个子图
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

for a_type in A_types:
    mask = (A_masked == a_type)
    B_values = B_masked[mask]
    B_values = B_values[~np.isnan(B_values)]  # 去除NaN值

    if len(B_values) > 0:
        # 计算直方图
        pixel_count, _ = np.histogram(B_values, bins=B_bins)

        # 转换为百分比
        total_pixels = np.sum(pixel_count)  # 总像素数量
        if total_pixels > 0:
            pixel_percentage = (pixel_count / total_pixels) * 100  # 百分比转换
        else:
            pixel_percentage = np.zeros_like(pixel_count)  # 防止除以零

        # 计算加权移动平均并绘制
        smooth_wma = weighted_moving_average(pixel_percentage, weights)
        axs.plot(B_bins[:-1], smooth_wma, label=label_mapping[a_type], color=colors[a_type])
        axs.set_title('Percentage of Total Pixels')

# 定义函数：获取二次拟合的方程和统计指标
def quadratic_fit_significance(B_values, D_values):
    B_squared = np.power(B_values, 2)
    X = np.column_stack((B_squared, B_values, np.ones_like(B_values)))
    model = sm.OLS(D_values, X).fit()
    return model.f_pvalue, model.fvalue, model.rsquared, model.params

axs.set_xlabel('Changes in cropland coverage from 2015 to 2022')
axs.set_ylabel('Percentage')
axs.legend()
plt.tight_layout()
# plt.savefig('slope_LPD_pixel_crop.png', dpi=600)
# plt.savefig('slope_LPD_pixel_tree.png', dpi=600)

plt.show()
plt.figure(figsize=(12, 5))

for a_type in A_types:
    mask = (A_masked == a_type)
    B_values = B_masked[mask]
    D_values = D_masked[mask]

    valid_mask = (~np.isnan(B_values)) & (~np.isnan(D_values))
    B_values = B_values[valid_mask]
    D_values = D_values[valid_mask]

    if len(B_values) > 0:
        plt.scatter(B_values, D_values, alpha=0.5, label=label_mapping[a_type], marker='o', color=colors[a_type])
        poly_fit = Polynomial.fit(B_values, D_values, 2)
        B_fit = np.linspace(np.min(B_values), np.max(B_values), 100)
        D_fit = poly_fit(B_fit)

        plt.plot(B_fit, D_fit, label=f'{label_mapping[a_type]} Quadratic Fit', linewidth=2, color=colors[a_type])
        D_fit_full = poly_fit(B_values)
        residuals = D_values - D_fit_full
        std_error = np.std(residuals)

        plt.fill_between(B_fit, D_fit - std_error, D_fit + std_error, alpha=0.2, color=colors[a_type])

        p_value, f_value, r_squared, params = quadratic_fit_significance(B_values, D_values)
        print(f'{label_mapping[a_type]} Fit: {params[0]:.4f}*B^2 + {params[1]:.4f}*B + {params[2]:.4f}')
        print(f'{label_mapping[a_type]} Significance: p-value = {p_value:.4e}, F-stat = {f_value:.4f}, R² = {r_squared:.4f}')

plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title('Relationship between RI and Slope by LPD Classes')
plt.xlabel('Changes in cropland coverage from 2015 to 2022')
plt.ylabel('RI')
plt.legend()
plt.tight_layout()
plt.show()



# -----------------------------------------------------------------------------------------------------------------------------------------
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\3_11\lpd_10000m\lpd_GGW_10000m.tif') as src_A:
    A = src_A.read(1)
    transform_A = src_A.transform
    height, width = A.shape

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\TREE_0.05\slope.tif') as src:
    B = src.read(1)

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\TREE_0.05\GGW_p_0.05.tif') as src:
    C = src.read(1)

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\3_11\GGW\GGW_tree.tif') as src:
    D = src.read(1)

# # 读取两个图像C1和C2
with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\CROP\GGW_p_0.05.tif') as src1:
    C1 = src1.read(1)

with rasterio.open(r'D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\5_21\LCC\TREE_0.05\GGW_p_0.05.tif') as src2:
    C2 = src2.read(1)

# 创建至少存在值的mask
mask_agg = (~np.isnan(C1)) | (~np.isnan(C2))

C = mask_agg

# 应用掩膜过滤
A_masked = np.where(C == 1, A, np.nan)
B_masked = np.where(C == 1, B, np.nan)
D_padded = np.pad(D, ((1, 1), (1, 1)), mode='constant', constant_values=np.nan)
D_masked = np.where(C == 1, D_padded, np.nan)


# 获取图像的行和列数
rows, cols = A.shape

# 获取每个像素的地理坐标（纬度和经度）
x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
latitudes, longitudes = rasterio.transform.xy(transform_A, y_coords, x_coords)
latitudes = np.array(latitudes)

A_types = [1, 5]
colors = {1: 'red', 5: 'green'}

B_bins = np.linspace(np.nanmin(B_masked), np.nanmax(B_masked), 100)
# Define the label mapping
label_mapping = {
    1: 'Declining',
    2: 'Early signs of decline',
    3: 'Stable but stressed',
    4: 'Stable and not stressed',
    5: 'Increasing'
}

# 定义加权移动平均方法
def weighted_moving_average(data, weights):
    return np.convolve(data, weights / weights.sum(), mode='same')

weights = np.array([1, 2, 3, 2, 1])  # 权重设置

# 创建一个子图
fig, axs = plt.subplots(1, 1, figsize=(7, 5))

for a_type in A_types:
    mask = (A_masked == a_type)
    B_values = B_masked[mask]
    B_values = B_values[~np.isnan(B_values)]  # 去除NaN值

    if len(B_values) > 0:
        # 计算直方图
        pixel_count, _ = np.histogram(B_values, bins=B_bins)

        # 转换为百分比
        total_pixels = np.sum(pixel_count)  # 总像素数量
        if total_pixels > 0:
            pixel_percentage = (pixel_count / total_pixels) * 100  # 百分比转换
        else:
            pixel_percentage = np.zeros_like(pixel_count)  # 防止除以零

        # 计算加权移动平均并绘制
        smooth_wma = weighted_moving_average(pixel_percentage, weights)
        axs.plot(B_bins[:-1], smooth_wma, label=label_mapping[a_type], color=colors[a_type])
        axs.set_title('Percentage of Total Pixels')

# 定义函数：获取二次拟合的方程和统计指标
def quadratic_fit_significance(B_values, D_values):
    B_squared = np.power(B_values, 2)
    X = np.column_stack((B_squared, B_values, np.ones_like(B_values)))
    model = sm.OLS(D_values, X).fit()
    return model.f_pvalue, model.fvalue, model.rsquared, model.params

#Weighted Moving Average (Percentage)
# 设置全局属性
axs.set_xlabel('Changes in forestland coverage from 2015 to 2022')
axs.set_ylabel('Percentage')
axs.legend()
plt.tight_layout()
# plt.savefig('slope_LPD_pixel_crop.png', dpi=600)
# plt.savefig('slope_LPD_pixel_tree.png', dpi=600)

plt.show()

# 绘制A_types下的散点图及二次拟合
plt.figure(figsize=(12, 5))

for a_type in A_types:
    mask = (A_masked == a_type)
    B_values = B_masked[mask]
    D_values = D_masked[mask]

    valid_mask = (~np.isnan(B_values)) & (~np.isnan(D_values))
    B_values = B_values[valid_mask]
    D_values = D_values[valid_mask]

    if len(B_values) > 0:
        plt.scatter(B_values, D_values, alpha=0.5, label=label_mapping[a_type], marker='o', color=colors[a_type])
        poly_fit = Polynomial.fit(B_values, D_values, 2)
        B_fit = np.linspace(np.min(B_values), np.max(B_values), 100)
        D_fit = poly_fit(B_fit)

        plt.plot(B_fit, D_fit, label=f'{label_mapping[a_type]} Quadratic Fit', linewidth=2, color=colors[a_type])
        D_fit_full = poly_fit(B_values)
        residuals = D_values - D_fit_full
        std_error = np.std(residuals)

        plt.fill_between(B_fit, D_fit - std_error, D_fit + std_error, alpha=0.2, color=colors[a_type])

        p_value, f_value, r_squared, params = quadratic_fit_significance(B_values, D_values)
        print(f'{label_mapping[a_type]} Fit: {params[0]:.4f}*B^2 + {params[1]:.4f}*B + {params[2]:.4f}')
        print(f'{label_mapping[a_type]} Significance: p-value = {p_value:.4e}, F-stat = {f_value:.4f}, R² = {r_squared:.4f}')

plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title('Relationship between RI and Slope by LPD Classes')
plt.xlabel('Changes in forestland coverage from 2015 to 2022')
plt.ylabel('RI')
plt.legend()
plt.tight_layout()
plt.show()
