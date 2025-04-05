import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from reliability import Fitters
import seaborn as sns
import locale
import matplotlib.font_manager as fm

# 添加系统字体路径
font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统
# font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux系统
# font_path = '/System/Library/Fonts/PingFang.ttc'  # MacOS系统

# 设置字体
plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]

# 设置中文显示和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']  # 优先使用的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体
plt.rcParams['font.size'] = 12  # 增加字体大小以提高可读性
sns.set_style('whitegrid')

# 确保在生成图表之前设置本地化
locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')

# 1. 数据加载与预处理
data = {
    '组数': range(1, 21),
    '周期': [12, 10, 18, 14, 14, 16, 20, 15, 20, 16, 16, 15, 15, 14, 19, 12, 17, 20, 13, 17]
}
df = pd.DataFrame(data)
lifetimes = df['周期'].sort_values().values

print("原始数据:")
print(df)
print("\n排序后的寿命数据:")
print(lifetimes)

# 2. 描述性统计分析
def descriptive_stats(data):
    stats_dict = {
        '样本量': len(data),
        '平均寿命': np.mean(data),
        '中位寿命': np.median(data),
        '最小寿命': np.min(data),
        '最大寿命': np.max(data),
        '标准差': np.std(data, ddof=1),
        '偏度': stats.skew(data),
        '峰度': stats.kurtosis(data)
    }
    return pd.DataFrame(stats_dict, index=['值'])

desc_stats = descriptive_stats(lifetimes)
print("\n描述性统计:")
print(desc_stats)

# 3. 非参数可靠性分析
def nonparametric_reliability(data):
    n = len(data)
    unique_times = np.unique(data)
    results = []
    
    for t in unique_times:
        failures = np.sum(data == t)
        at_risk = np.sum(data >= t)
        prob_failure = failures / n
        cum_failure = np.sum(data <= t) / n
        # 保留上一时刻的可靠度
        last_reliability = reliability if results else 1
        reliability = 1 - cum_failure
        if at_risk > 0:
            hazard_rate = prob_failure / last_reliability
        else:
            hazard_rate = np.nan
        
        # 中位秩估计
        i = np.where(unique_times == t)[0][0] + 1
        median_rank = (i - 0.3) / (n + 0.4)
        
        results.append({
            '周期': t,
            '失效数': failures,
            '风险集': at_risk,
            'f(t)': prob_failure,
            'F(t)': cum_failure,
            'R(t)': reliability,
            'λ(t)': hazard_rate,
            '中位秩F(t)': median_rank,
            '中位秩R(t)': 1 - median_rank
        })
    
    return pd.DataFrame(results)

nonparam_results = nonparametric_reliability(lifetimes)
print("\n非参数可靠性分析结果:")
print(nonparam_results)


# 4. 威布尔分布拟合
def weibull_fit(data):
    weibull_fit = Fitters.Fit_Weibull_2P(failures=data, show_probability_plot=False, print_results=False)
    beta = weibull_fit.beta
    alpha = weibull_fit.alpha  # 新版本中使用alpha而不是eta
    return beta, alpha, weibull_fit

beta, alpha, weibull_model = weibull_fit(lifetimes)
print(f"\n威布尔分布拟合结果: 形状参数β={beta:.2f}, 尺度参数α={alpha:.2f}")

def plot_combined_curves_with_connections(nonparam_results, beta, alpha):
    """在同一图中绘制四条曲线：经验值带折线，威布尔拟合用虚线"""
    plt.figure(figsize=(12, 7))
    # 设置全局字体属性
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    t_empirical = nonparam_results['周期']
    t_smooth = np.linspace(min(t_empirical)-1, max(t_empirical)+1, 200)
    
    # 计算威布尔拟合曲线（虚线）
    f_fit = (beta/alpha) * (t_smooth/alpha)**(beta-1) * np.exp(-(t_smooth/alpha)**beta)
    F_fit = 1 - np.exp(-(t_smooth/alpha)**beta)
    R_fit = np.exp(-(t_smooth/alpha)**beta)
    lambda_fit = (beta/alpha) * (t_smooth/alpha)**(beta-1)
    
    # 绘制经验数据（散点+折线）
    plt.plot(t_empirical, nonparam_results['f(t)'], 'bo-', linewidth=1.5, markersize=8, 
             label='经验概率密度 f(t)')
    plt.plot(t_empirical, nonparam_results['F(t)'], 'g^-', linewidth=1.5, markersize=8, 
             label='经验累积失效 F(t)')
    plt.plot(t_empirical, nonparam_results['R(t)'], 'rs-', linewidth=1.5, markersize=8, 
             label='经验可靠性 R(t)')
    valid_idx = ~np.isnan(nonparam_results['λ(t)'])
    plt.plot(t_empirical[valid_idx], nonparam_results['λ(t)'][valid_idx], 'mD-', 
             linewidth=1.5, markersize=8, label='经验故障率 λ(t)')
    
    # 绘制威布尔拟合曲线（虚线）
    plt.plot(t_smooth, f_fit, 'b--', linewidth=2, alpha=0.7, 
             label='威布尔拟合 f(t)')
    plt.plot(t_smooth, F_fit, 'g--', linewidth=2, alpha=0.7, 
             label='威布尔拟合 F(t)')
    plt.plot(t_smooth, R_fit, 'r--', linewidth=2, alpha=0.7, 
             label='威布尔拟合 R(t)')
    plt.plot(t_smooth, lambda_fit, 'm--', linewidth=2, alpha=0.7, 
             label='威布尔拟合 λ(t)')
    
    # 图表装饰
    plt.xlabel('周期', fontsize=12)
    plt.ylabel('函数值', fontsize=12)
    plt.title('可靠性函数对比：经验值（实线+标记） vs 威布尔拟合（虚线）', fontsize=14, pad=20)
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), framealpha=1)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 调整坐标轴范围
    y_max = max(np.nanmax(nonparam_results['f(t)']), 
                np.nanmax(nonparam_results['F(t)']),
                np.nanmax(nonparam_results['R(t)']),
                np.nanmax(nonparam_results['λ(t)'][valid_idx])) * 1.1
    plt.ylim(-0.05, y_max)
    
    plt.tight_layout()
    plt.show()

# 调用函数
plot_combined_curves_with_connections(nonparam_results, beta, alpha)

# 5. 可靠性指标计算
def reliability_metrics(beta, alpha):
    B10 = alpha * (-np.log(0.9))**(1/beta)
    B50 = alpha * (-np.log(0.5))**(1/beta)
    return {
        'B10寿命(10%失效)': B10,
        'B50寿命(中位寿命)': B50,
        '特征寿命(α)': alpha,
        '形状参数(β)': beta
    }

metrics = reliability_metrics(beta, alpha)
print("\n关键可靠性指标:")
print(pd.DataFrame(metrics, index=['值']))

# 6. 可视化分析
def plot_six_subplots_optimized(nonparam_results, beta, alpha):
    """优化后的六子图布局，确保所有内容可见"""
    # 创建更大的画布并调整子图间距
    plt.figure(figsize=(18, 22))  # 宽度增加，高度大幅增加
    plt.rcParams['font.size'] = 12  # 调大全局字体
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 生成平滑时间轴
    t_smooth = np.linspace(min(lifetimes)-1, max(lifetimes)+1, 200)
    
    # ===== 上排：f(t) 和 λ(t) =====
    # 1. 概率密度函数 f(t)
    ax1 = plt.subplot(3, 2, 1)
    plt.plot(nonparam_results['周期'], nonparam_results['f(t)'], 'bo-', 
             markersize=6, linewidth=1.2, label='经验值')
    plt.plot(t_smooth, (beta/alpha)*(t_smooth/alpha)**(beta-1)*np.exp(-(t_smooth/alpha)**beta), 
             'r--', linewidth=1.8, label='威布尔拟合')
    plt.xlabel('周期', fontsize=11)  # 调小坐标轴字体
    plt.ylabel('概率密度 f(t)', fontsize=11)
    plt.title('(a) 概率密度函数', y=1, loc='left', fontsize=12)  # 调整标题位置
    plt.legend(fontsize=10, bbox_to_anchor=(0.7, 1))  # 移动图例位置
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 2. 故障率函数 λ(t)
    ax2 = plt.subplot(3, 2, 2)
    valid_idx = ~np.isnan(nonparam_results['λ(t)'])
    plt.plot(nonparam_results['周期'][valid_idx], nonparam_results['λ(t)'][valid_idx], 
             'mo-', markersize=6, linewidth=1.2, label='经验值')
    plt.plot(t_smooth, (beta/alpha)*(t_smooth/alpha)**(beta-1), 'r--', 
             linewidth=1.8, label='威布尔拟合')
    plt.xlabel('周期', fontsize=11)
    plt.ylabel('故障率 λ(t)', fontsize=11)
    plt.title('(b) 故障率函数', y=1, loc='left', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(0.7, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # ===== 中排：F(t) 和 R(t) =====
    # 3. 累积失效分布 F(t)
    ax3 = plt.subplot(3, 2, 3)
    plt.plot(nonparam_results['周期'], nonparam_results['F(t)'], 'go-', 
             markersize=6, linewidth=1.2, label='经验值')
    plt.plot(t_smooth, 1 - np.exp(-(t_smooth/alpha)**beta), 'r--', 
             linewidth=1.8, label='威布尔拟合')
    plt.xlabel('周期', fontsize=11)
    plt.ylabel('累积失效 F(t)', fontsize=11)
    plt.title('(c) 累积失效分布', y=1, loc='left', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(0.7, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 4. 可靠性函数 R(t)
    ax4 = plt.subplot(3, 2, 4)
    plt.plot(nonparam_results['周期'], nonparam_results['R(t)'], 'rs-', 
             markersize=6, linewidth=1.2, label='经验值')
    plt.plot(t_smooth, np.exp(-(t_smooth/alpha)**beta), 'r--', 
             linewidth=1.8, label='威布尔拟合')
    plt.xlabel('周期', fontsize=11)
    plt.ylabel('可靠度 R(t)', fontsize=11)
    plt.title('(d) 可靠性函数', y=1, loc='left', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(0.7, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # ===== 下排：中位秩F(t)和R(t) =====
    # 5. 中位秩累积失效 F(t)
    ax5 = plt.subplot(3, 2, 5)
    plt.plot(nonparam_results['周期'], nonparam_results['中位秩F(t)'], 'yo-', 
             markersize=6, linewidth=1.2, label='中位秩F(t)')
    plt.plot(t_smooth, 1 - np.exp(-(t_smooth/alpha)**beta), 'r--', 
             linewidth=1.8, label='威布尔拟合')
    plt.xlabel('周期', fontsize=11)
    plt.ylabel('中位秩 F(t)', fontsize=11)
    plt.title('(e) 中位秩累积失效', y=1, loc='left', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(0.7, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 6. 中位秩可靠性 R(t)
    ax6 = plt.subplot(3, 2, 6)
    plt.plot(nonparam_results['周期'], nonparam_results['中位秩R(t)'], 'co-', 
             markersize=6, linewidth=1.2, label='中位秩R(t)')
    plt.plot(t_smooth, np.exp(-(t_smooth/alpha)**beta), 'r--', 
             linewidth=1.8, label='威布尔拟合')
    plt.xlabel('周期', fontsize=11)
    plt.ylabel('中位秩 R(t)', fontsize=11)
    plt.title('(f) 中位秩可靠性', y=1, loc='left', fontsize=12)
    plt.legend(fontsize=10, bbox_to_anchor=(0.7, 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 全局调整
    plt.subplots_adjust(
        left=0.1,      # 左边距
        right=0.9,     # 右边距
        bottom=0.08,   # 底边距
        top=0.92,      # 顶部空间
        wspace=0.3,    # 子图水平间距
        hspace=0.4     # 子图垂直间距
    )
    # plt.suptitle('可靠性分析全函数对比', y=0.95, fontsize=14)
    plt.show()

# 调用函数
plot_six_subplots_optimized(nonparam_results, beta, alpha)

# 7. 威布尔概率图
def plot_weibull_probability(data, beta, alpha):
    plt.figure(figsize=(8, 6))
    # 设置全局字体属性
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 计算经验累积分布函数点
    n = len(data)
    sorted_data = np.sort(data)
    median_ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)  # 中位秩
    
    # 计算威布尔概率图的坐标
    x = np.log(sorted_data)
    y = np.log(-np.log(1 - median_ranks))
    
    # 绘制数据点
    plt.scatter(x, y, marker='o', label=u'数据点')
    
    # 计算拟合线
    x_fit = np.log(np.linspace(min(data), max(data), 100))
    y_fit = beta * (x_fit - np.log(alpha))
    
    # 绘制拟合线
    plt.plot(x_fit, y_fit, 'r-', label=u'拟合线')
    
    plt.title(u'威布尔概率图')
    plt.xlabel(u'ln(t)')
    plt.ylabel(u'ln(-ln(1-F(t)))')
    plt.grid(True)
    plt.legend()
    plt.show()

# 替换原来的威布尔概率图绘制代码
plot_weibull_probability(lifetimes, beta, alpha)

# 8. 保存结果
output = pd.concat([
    desc_stats.T,
    pd.DataFrame(metrics, index=['值']).T,
    nonparam_results
], axis=1)

output.to_excel('可靠性分析结果.xlsx', index=False)
print("\n分析结果已保存到 '可靠性分析结果.xlsx'")