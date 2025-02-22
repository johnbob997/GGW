import rasterio
import numpy as np

import rasterio
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # 导入 tqdm 库

# Define the value range of A and B
a_range = [1, 2, 3]            # The valid value range of the grid A
b_range = [1, 2, 3, 4, 5]      # The valid value range of the grid B

def count_value_pairs_sparse(a_raster_path, b_raster_path):
    pair_counts = defaultdict(int)

    with rasterio.open(a_raster_path) as a_src, rasterio.open(b_raster_path) as b_src:
        assert a_src.shape == b_src.shape, "The sizes of A and B grids are inconsistent"

        total_blocks = sum(1 for _ in a_src.block_windows(1))

        with tqdm(total=total_blocks, desc="Processing raster blocks", unit="block") as pbar:
            for ji, window in a_src.block_windows(1):
                a_data = a_src.read(1, window=window)
                b_data = b_src.read(1, window=window)


                a_nodata = a_src.nodatavals[0]
                b_nodata = b_src.nodatavals[0]


                valid_mask = (a_data != a_nodata) & (b_data != b_nodata)
                a_valid = a_data[valid_mask]
                b_valid = b_data[valid_mask]


                for a_val, b_val in zip(a_valid, b_valid):
                    pair_counts[(a_val, b_val)] += 1


                pbar.update(1)

    return pair_counts

a_raster_path = "D:/postgraduate_file/GGW_23.7.11-/RESULT_GGW/3_11/NDVI_MK/TrajectCalNDVI_MK.tif"  # 替换为 A 栅格路径
b_raster_path = "D:/postgraduate_file/GGW_23.7.11-/RESULT_GGW/3_11/LPD_FINAL/LPD_30M_GGW.tif"  # 替换为 B 栅格路径

result = count_value_pairs_sparse(a_raster_path, b_raster_path)
print("Statistical results dictionary：")
print(result)



# def count_value_pairs_fast(a_raster_path, b_raster_path, a_range, b_range):
#     # 打开栅格文件
#     with rasterio.open(a_raster_path) as a_src, rasterio.open(b_raster_path) as b_src:
#         if a_src.shape != b_src.shape:
#             raise ValueError("两个栅格的形状不一致，请确保它们具有相同的分辨率和范围！")
#
#         # 读取栅格数据
#         a_data = a_src.read(1)
#         b_data = b_src.read(1)
#
#         # 获取无效值
#         a_nodata = a_src.nodatavals[0]
#         b_nodata = b_src.nodatavals[0]
#
#         # 替换无效值为 NaN
#         a_data = np.where(a_data == a_nodata, np.nan, a_data)
#         b_data = np.where(b_data == b_nodata, np.nan, b_data)
#
#         # 创建掩码，仅保留有效值
#         mask = ~np.isnan(a_data) & ~np.isnan(b_data)
#         a_data = a_data[mask].astype(int)
#         b_data = b_data[mask].astype(int)
#
#         # 使用 np.histogram2d 统计值对的频率
#         count_matrix, _, _ = np.histogram2d(
#             a_data, b_data, bins=[np.arange(min(a_range) - 0.5, max(a_range) + 1),
#                                   np.arange(min(b_range) - 0.5, max(b_range) + 1)]
#         )
#
#     return count_matrix
