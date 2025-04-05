import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.stats import weibull_min
import seaborn as sns

# 在导入库之后添加以下代码
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']  # 优先使用的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体

# 数据
data = np.array([12,10,18,14,14,16,20,15,20,16,16,15,15,14,19,12,17,20,13,17])

def basic_reliability_metrics(data):
    """计算基础可靠性指标"""
    mttf = np.mean(data)
    std = np.std(data)
    
    # 排序数据用于计算B10寿命
    sorted_data = np.sort(data)
    b10_index = int(0.1 * len(data))
    b10_life = sorted_data[b10_index]
    
    return {
        'MTTF': mttf,
        'Standard Deviation': std,
        'B10 Life': b10_life
    }

def bootstrap_analysis(data, n_bootstrap=1000):
    """使用Bootstrap方法进行不确定性分析"""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        'Bootstrap Mean': np.mean(bootstrap_means),
        'CI Lower': confidence_interval[0],
        'CI Upper': confidence_interval[1]
    }

def plot_survival_curve(data):
    """绘制Kaplan-Meier生存曲线"""
    kmf = KaplanMeierFitter()
    # 创建事件数据（这里假设所有数据点都是失效数据）
    events = np.ones(len(data))
    kmf.fit(data, events, label='Kaplan-Meier Estimate')
    
    plt.figure(figsize=(10, 6))
    kmf.plot()
    plt.title('Kaplan-Meier生存曲线')
    plt.xlabel('时间')
    plt.ylabel('生存概率')
    plt.grid(True)
    plt.show()

def weibull_analysis(data):
    """进行Weibull分布分析"""
    # 使用最大似然估计法拟合Weibull分布
    shape, loc, scale = weibull_min.fit(data, floc=0)
    
    return {
        'Shape Parameter (β)': shape,
        'Scale Parameter (η)': scale
    }

def calculate_failure_rate(data):
    """计算故障率"""
    sorted_data = np.sort(data)
    n = len(data)
    failure_rates = []
    times = []
    
    for i in range(n-1):
        # 当前时间点
        t = sorted_data[i]
        # 存活数（大于当前时间的数量）
        survivors = np.sum(data > t)
        # 在此时间点的失效数
        failures = np.sum(data == t)
        
        if survivors > 0:
            failure_rate = failures / survivors
            failure_rates.append(failure_rate)
            times.append(t)
    
    return times, failure_rates

def plot_failure_rate(data):
    """绘制故障率曲线"""
    times, failure_rates = calculate_failure_rate(data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, failure_rates, 'bo-')
    plt.title('故障率曲线')
    plt.xlabel('时间')
    plt.ylabel('故障率')
    plt.grid(True)
    plt.show()

def bayesian_estimation(data, prior_mean=15, prior_std=3):
    """贝叶斯估计（假设正态分布）"""
    n = len(data)
    sample_mean = np.mean(data)
    sample_var = np.var(data)
    
    # 先验精度和数据精度
    prior_precision = 1 / (prior_std ** 2)
    likelihood_precision = n / sample_var
    
    # 后验分布参数
    posterior_precision = prior_precision + likelihood_precision
    posterior_mean = (prior_mean * prior_precision + sample_mean * likelihood_precision) / posterior_precision
    posterior_std = np.sqrt(1 / posterior_precision)
    
    return {
        'Posterior Mean': posterior_mean,
        'Posterior Std': posterior_std,
        'CI Lower': posterior_mean - 1.96 * posterior_std,
        'CI Upper': posterior_mean + 1.96 * posterior_std
    }

def plot_distributions(data):
    """绘制Bootstrap分布直方图和Weibull概率图"""
    plt.figure(figsize=(12, 6))
    
    # Bootstrap分布
    bootstrap_means = []
    for _ in range(1000):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    plt.subplot(1, 2, 1)
    sns.histplot(bootstrap_means, kde=True)
    plt.title('Bootstrap分布')
    plt.xlabel('MTTF估计值')
    plt.ylabel('频数')
    
    # Weibull概率图
    plt.subplot(1, 2, 2)
    # 首先估计Weibull参数
    shape, loc, scale = weibull_min.fit(data, floc=0)
    # 使用估计的形状参数进行概率图绘制
    stats.probplot(data, dist="weibull_min", sparams=(shape,), plot=plt)
    plt.title('Weibull概率图')
    
    plt.tight_layout()
    plt.show()

# 执行分析
basic_metrics = basic_reliability_metrics(data)
bootstrap_results = bootstrap_analysis(data)
weibull_params = weibull_analysis(data)

# 打印结果
print("\n基础可靠性指标:")
print(f"MTTF: {basic_metrics['MTTF']:.2f}")
print(f"标准差: {basic_metrics['Standard Deviation']:.2f}")
print(f"B10寿命: {basic_metrics['B10 Life']:.2f}")

print("\nBootstrap分析结果:")
print(f"Bootstrap平均值: {bootstrap_results['Bootstrap Mean']:.2f}")
print(f"95%置信区间: ({bootstrap_results['CI Lower']:.2f}, {bootstrap_results['CI Upper']:.2f})")

print("\nWeibull分布参数:")
print(f"形状参数(β): {weibull_params['Shape Parameter (β)']:.2f}")
print(f"尺度参数(η): {weibull_params['Scale Parameter (η)']:.2f}")

# 绘制生存曲线
plot_survival_curve(data)

# 执行新增的分析
bayesian_results = bayesian_estimation(data)

# 打印新增的结果
print("\n贝叶斯估计结果:")
print(f"后验均值: {bayesian_results['Posterior Mean']:.2f}")
print(f"后验标准差: {bayesian_results['Posterior Std']:.2f}")
print(f"95%置信区间: ({bayesian_results['CI Lower']:.2f}, {bayesian_results['CI Upper']:.2f})")

# 绘制新增的图表
plot_failure_rate(data)
plot_distributions(data)
