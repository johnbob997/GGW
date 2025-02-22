import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

# 设置默认字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
# 读取数据
# df = pd.read_excel('D:\postgraduate file\GGW_23.7.11-\RESULT_GGW\\3_11\\Sample_C.xlsx', names=['VAR', 'Declining', 'Increasing'])
df = pd.read_excel('D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\\3_11\\Sample_C_NEW.xlsx', names=['VAR', 'Declining', 'Increasing'])
plt.figure(figsize=(12, 7))
# 将'DE'和'IN'列合并成一个DataFrame，并创建一个新的列来标识数据来源
df_melted = df.melt(id_vars=['VAR'], var_name='Source', value_name='Values')
print(df_melted)
df_melted['Values'] = df_melted['Values'].multiply(100)
print(df_melted)
# 使用Seaborn绘制分组箱线图
ax = sns.boxplot(data=df_melted, x='VAR', y='Values', hue='Source',palette={'Declining': '#fe0000', 'Increasing': '#55ff00'})

# plt.xlabel('Variables')
plt.ylabel('RI Value (%)')
# plt.title('Grouped Boxplot of DE and IN Values by VAR')
# 设置图例并取消边框，并修改文字内容

# 设置图例并取消边框，并修改文字内容
legend = plt.legend(frameon=False)
# 设置图例位置
legend.set_bbox_to_anchor((0.5, 0.8))

# 取消右边和上边的边界线
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# 取消横坐标最下方的VAR注释
ax.set_xlabel('')

def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta for two independent samples.

    Parameters:
    x : array-like
        First sample data.
    y : array-like
        Second sample data.

    Returns:
    delta : float
        Cliff's Delta value.
    """
    n_x = len(x)
    n_y = len(y)
    ranks = mannwhitneyu(x, y, alternative='two-sided').statistic
    delta = (2 * ranks - n_x * n_y) / (n_x * n_y)
    return delta
# 对各VAR的IN和DE两种情况做MWU检验
vars = df_melted['VAR'].unique()
for var in vars:
    de_values = df[df['VAR'] == var]['Declining'].dropna()
    in_values = df[df['VAR'] == var]['Increasing'].dropna()
    stat, p_value = mannwhitneyu(de_values, in_values, alternative='two-sided')
    print(f"Mann-Whitney U test for {var}: p-value = {p_value}")
    print(cliffs_delta(de_values,in_values))
    abs_delta = abs(cliffs_delta(de_values,in_values))
    if abs_delta < 0.11:
        print("⊘")  # Not significant
    elif 0.11 <= abs_delta < 0.28:
        print("△")  # Small effect
    elif 0.28 <= abs_delta < 0.43:
        print("▲")  # Medium effect
    elif abs_delta >= 0.43:
        print("■")  # Large effect

# plt.savefig('NEW_MW.png', dpi=600)
plt.ylim(0,100)
plt.show()


# df = pd.read_excel('D:\postgraduate_file\GGW_23.7.11-\RESULT_GGW\\3_11\\Sample_C_AGG.xlsx', names=['VAR', 'Declining', 'Increasing'])
# plt.figure(figsize=(5, 7))
# # 将'DE'和'IN'列合并成一个DataFrame，并创建一个新的列来标识数据来源
# df_melted = df.melt(id_vars=['VAR'], var_name='Source', value_name='Values')
# print(df_melted)
# df_melted['Values'] = df_melted['Values'].multiply(100)
# print(df_melted)
# # # 使用Seaborn绘制分组箱线图
# ax = sns.boxplot(data=df_melted, x='VAR', y='Values', hue='Source',palette={'Declining': '#fe0000', 'Increasing': '#55ff00'})
#
#
# # plt.xlabel('Variables')
# plt.ylabel('RI Value (%)')
# # plt.title('Grouped Boxplot of DE and IN Values by VAR')
# # 设置图例并取消边框，并修改文字内容
#
# # 设置图例并取消边框，并修改文字内容
# legend = plt.legend(frameon=False)
# # 设置图例位置
# legend.set_bbox_to_anchor((0.5, 0.8))
#
# # 取消右边和上边的边界线
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # 取消横坐标最下方的VAR注释
# ax.set_xlabel('')
#
# def cliffs_delta(x, y):
#     """
#     Calculate Cliff's Delta for two independent samples.
#
#     Parameters:
#     x : array-like
#         First sample data.
#     y : array-like
#         Second sample data.
#
#     Returns:
#     delta : float
#         Cliff's Delta value.
#     """
#     n_x = len(x)
#     n_y = len(y)
#     ranks = mannwhitneyu(x, y, alternative='two-sided').statistic
#     delta = (2 * ranks - n_x * n_y) / (n_x * n_y)
#     return delta
# # 对各VAR的IN和DE两种情况做MWU检验
# vars = df_melted['VAR'].unique()
# for var in vars:
#     de_values = df[df['VAR'] == var]['Declining'].dropna()
#     in_values = df[df['VAR'] == var]['Increasing'].dropna()
#     stat, p_value = mannwhitneyu(de_values, in_values, alternative='two-sided')
#     print(f"Mann-Whitney U test for {var}: p-value = {p_value}")
#     print(cliffs_delta(de_values,in_values))
#     abs_delta = abs(cliffs_delta(de_values,in_values))
#     if abs_delta < 0.11:
#         print("⊘")  # Not significant
#     elif 0.11 <= abs_delta < 0.28:
#         print("△")  # Small effect
#     elif 0.28 <= abs_delta < 0.43:
#         print("▲")  # Medium effect
#     elif abs_delta >= 0.43:
#         print("■")  # Large effect
#
# # plt.savefig('MW_AGG_3.png', dpi=600)
# plt.ylim(0,100)
# plt.show()



