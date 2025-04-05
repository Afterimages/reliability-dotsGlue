import numpy as np
import matplotlib.pyplot as plt
from reliability.Distributions import Weibull_Distribution
from reliability.Fitters import Fit_Weibull_2P
from sklearn.utils import resample

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 原始数据
lifetimes = np.array([12, 10, 18, 14, 14, 16, 20, 15, 20, 16, 
                     16, 15, 15, 14, 19, 12, 17, 20, 13, 17])

# 1. 首先拟合威布尔分布
fit = Fit_Weibull_2P(failures=lifetimes, show_probability_plot=False, print_results=False)
alpha, beta = fit.alpha, fit.beta
print(f"威布尔拟合结果: α={alpha:.2f}, β={beta:.2f}")

# 2. 修正后的蒙特卡洛仿真
def monte_carlo_simulation(alpha, beta, n_sim=1000, sample_size=20):
    """修正后的仿真函数"""
    simulated_b10 = []
    simulated_alpha = []
    simulated_beta = []
    
    dist = Weibull_Distribution(alpha=alpha, beta=beta)
    
    for _ in range(n_sim):
        # 修正点：使用正确参数生成样本
        virtual_data = dist.random_samples(number_of_samples=sample_size)  # 关键修正
        
        try:
            fit = Fit_Weibull_2P(failures=virtual_data, show_probability_plot=False, print_results=False)
            b10 = fit.alpha * (-np.log(0.9))**(1/fit.beta)
            simulated_b10.append(b10)
            simulated_alpha.append(fit.alpha)
            simulated_beta.append(fit.beta)
        except:
            continue
    
    return np.array(simulated_b10), np.array(simulated_alpha), np.array(simulated_beta)

# 运行蒙特卡洛仿真
sim_b10, sim_alpha, sim_beta = monte_carlo_simulation(alpha, beta, n_sim=5000)

# 可视化蒙特卡洛结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(sim_b10, bins=30, density=True, alpha=0.7, color='skyblue')
plt.axvline(np.percentile(sim_b10, 50), color='r', linestyle='--', label='中位数')
plt.xlabel('B10寿命')
plt.title('B10寿命分布(蒙特卡洛)')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(sim_alpha, bins=30, density=True, alpha=0.7, color='orange')
plt.axvline(alpha, color='k', linestyle='-', label='原始α')
plt.xlabel('尺度参数α')
plt.title('α参数分布')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(sim_beta, bins=30, density=True, alpha=0.7, color='green')
plt.axvline(beta, color='k', linestyle='-', label='原始β')
plt.xlabel('形状参数β')
plt.title('β参数分布')
plt.legend()

plt.tight_layout()
plt.suptitle('蒙特卡洛仿真结果', y=1.02)
plt.show()

# 3. Bootstrap分析
def bootstrap_analysis(data, n_iter=1000):
    b10_bs, alpha_bs, beta_bs = [], [], []
    
    for _ in range(n_iter):
        sample = resample(data)
        try:
            fit = Fit_Weibull_2P(failures=sample, show_probability_plot=False, print_results=False)
            b10 = fit.alpha * (-np.log(0.9))**(1/fit.beta)
            b10_bs.append(b10)
            alpha_bs.append(fit.alpha)
            beta_bs.append(fit.beta)
        except:
            continue
    
    return np.array(b10_bs), np.array(alpha_bs), np.array(beta_bs)

b10_bs, alpha_bs, beta_bs = bootstrap_analysis(lifetimes)

# 可视化Bootstrap结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(b10_bs, bins=30, density=True, alpha=0.7, color='skyblue')
plt.axvline(np.percentile(b10_bs, 50), color='r', linestyle='--', label='中位数')
plt.xlabel('B10寿命')
plt.title('Bootstrap B10分布')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(alpha_bs, bins=30, density=True, alpha=0.7, color='orange')
plt.axvline(alpha, color='k', linestyle='-', label='原始α')
plt.xlabel('尺度参数α')
plt.title('Bootstrap α分布')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(beta_bs, bins=30, density=True, alpha=0.7, color='green')
plt.axvline(beta, color='k', linestyle='-', label='原始β')
plt.xlabel('形状参数β')
plt.title('Bootstrap β分布')
plt.legend()

plt.tight_layout()
plt.suptitle('Bootstrap分析结果', y=1.02)
plt.show()

# 4. 结果对比
print("\n关键结果对比：")
print(f"原始点估计: B10={alpha * (-np.log(0.9))**(1/beta):.2f}, α={alpha:.2f}, β={beta:.2f}")
print(f"蒙特卡洛90% CI: B10=[{np.percentile(sim_b10, 5):.2f}, {np.percentile(sim_b10, 95):.2f}],
      α=[{np.percentile(sim_alpha, 5):.2f}, {np.percentile(sim_alpha, 95):.2f}],
      β=[{np.percentile(sim_beta, 5):.2f}, {np.percentile(sim_beta, 95):.2f}]")
print(f"Bootstrap95% CI: B10=[{np.percentile(b10_bs, 2.5):.2f}, {np.percentile(b10_bs, 97.5):.2f}],
      α=[{np.percentile(alpha_bs, 2.5):.2f}, {np.percentile(alpha_bs, 97.5):.2f}],
      β=[{np.percentile(beta_bs, 2.5):.2f}, {np.percentile(beta_bs, 97.5):.2f}]")

print("\n=== 可靠性分析最终结果 ===")
print(f"{'指标':<15}{'原始估计':<15}{'蒙特卡洛(90% CI)':<25}{'Bootstrap(95% CI)':<25}")
print("-" * 70)

# B10寿命行（修正关键错误）
print(f"{'B10寿命':<15}{alpha * (-np.log(0.9))**(1/beta):.2f}{'':<13}"
      f"[{np.percentile(sim_b10, 5):.2f}, {np.percentile(sim_b10, 95):.2f}]{'':<10}"
      f"[{np.percentile(b10_bs, 2.5):.2f}, {np.percentile(b10_bs, 97.5):.2f}]")

# 其他参数行
print(f"{'尺度参数α':<15}{alpha:.2f}{'':<13}"
      f"[{np.percentile(sim_alpha, 5):.2f}, {np.percentile(sim_alpha, 95):.2f}]{'':<10}"
      f"[{np.percentile(alpha_bs, 2.5):.2f}, {np.percentile(alpha_bs, 97.5):.2f}]")

print(f"{'形状参数β':<15}{beta:.2f}{'':<13}"
      f"[{np.percentile(sim_beta, 5):.2f}, {np.percentile(sim_beta, 95):.2f}]{'':<10}"
      f"[{np.percentile(beta_bs, 2.5):.2f}, {np.percentile(beta_bs, 97.5):.2f}]")


# 5. 保存关键数据
results = {
    'MC_B10_mean': np.mean(sim_b10),
    'MC_B10_CI': [np.percentile(sim_b10, 5), np.percentile(sim_b10, 95)],
    'BS_B10_CI': [np.percentile(b10_bs, 2.5), np.percentile(b10_bs, 97.5)],
    'Original_alpha': alpha,
    'Original_beta': beta
}

import pandas as pd
pd.DataFrame(results, index=['值']).to_excel('可靠性分析结果.xlsx')
print("\n分析结果已保存到Excel文件")