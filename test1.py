import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.stats import weibull_min
from sklearn.utils import resample

# 设置中文显示和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 输入数据（失效循环次数）
failure_cycles = np.array([12,10,18,14,14,16,20,15,20,16,16,15,15,14,19,12,17,20,13,17])

# ======================
# 1. 基础可靠性指标计算
# ======================

# Kaplan-Meier可靠度估计
kmf = KaplanMeierFitter()
kmf.fit(failure_cycles, event_observed=np.ones_like(failure_cycles))
survival_prob = kmf.survival_function_
survival_prob['故障率'] = 1 - survival_prob['KM_estimate']

# 平均失效前时间(MTTF)
mttf = np.mean(failure_cycles)

# B10寿命（可靠度为90%时的时间）
b10_life = survival_prob.index[np.where(survival_prob['KM_estimate'] <= 0.9)[0][0]]

# ======================
# 2. Weibull分布拟合
# ======================

# 拟合Weibull分布参数
shape, loc, scale = weibull_min.fit(failure_cycles, floc=0)
beta, eta = shape, scale

# 生成Weibull预测曲线
x = np.linspace(0, max(failure_cycles), 100)
weibull_pdf = weibull_min.pdf(x, beta, scale=eta)
weibull_cdf = weibull_min.cdf(x, beta, scale=eta)

# ======================
# 3. Bootstrap置信区间
# ======================

# 重采样1000次计算MTTF的置信区间
# 1. Bootstrap重采样生成1000组虚拟数据
np.random.seed(42)
bootstrap_samples = np.random.choice(failure_cycles, size=(1000, len(failure_cycles)), replace=True)
bootstrap_mttf = np.mean(bootstrap_samples, axis=1)
ci_low, ci_high = np.percentile(bootstrap_mttf, [2.5, 97.5])

# ======================
# 4. 可视化
# ======================

fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# 1. Kaplan-Meier生存曲线
kmf.plot(ax=ax[0][0])  # 不使用plot_survival_function
ax[0][0].axhline(y=0.9, color='r', linestyle='--', label='B10寿命(90%可靠度)')
ax[0][0].set_title('Kaplan-Meier生存曲线')
ax[0][0].set_xlabel('循环次数')
ax[0][0].set_ylabel('可靠度R(N)')
ax[0][0].legend()

# 4.2 Weibull分布拟合验证
ax[0, 1].plot(x, weibull_cdf, label='Weibull CDF')
ecdf = np.arange(1, len(failure_cycles)+1)/len(failure_cycles)  # 经验CDF
ax[0, 1].step(np.sort(failure_cycles), ecdf, where='post', label='实际数据')
ax[0, 1].set_title('Weibull分布验证')
ax[0, 1].legend()

# 4.3 故障率曲线
failure_rates = []
unique_times = np.unique(failure_cycles)
for n in unique_times:
    at_risk = len(failure_cycles[failure_cycles >= n])
    failed = len(failure_cycles[failure_cycles == n])
    failure_rates.append(failed / at_risk if at_risk > 0 else 0)
ax[1, 0].plot(unique_times, failure_rates, 'o-', label='故障率')
ax[1, 0].set_title('故障率 λ(t)')
ax[1, 0].set_xlabel('循环次数')

# 4.4 MTTF的Bootstrap分布
ax[1, 1].hist(bootstrap_mttf, bins=30, density=True, alpha=0.7)
ax[1, 1].axvline(mttf, color='r', label=f'MTTF={mttf:.1f}')
ax[1, 1].axvspan(ci_low, ci_high, color='green', alpha=0.2, 
                 label=f'95% CI: [{ci_low:.1f}, {ci_high:.1f}]')
ax[1, 1].set_title('MTTF的Bootstrap估计')
ax[1, 1].legend()

plt.tight_layout()
plt.savefig('reliability_analysis.png', dpi=300)
plt.show()

# ======================
# 5. 输出分析报告
# ======================

report = f"""
可靠性分析报告
================================
1. 基础指标
   - 平均失效前时间 (MTTF): {mttf:.1f} 次循环 (95% CI: {ci_low:.1f}~{ci_high:.1f})
   - B10寿命 (10%失效概率): {b10_life} 次循环
   - 最高故障率: {max(failure_rates):.1%} (发生在 {survival_prob.index[np.argmax(failure_rates)]} 次循环)

2. Weibull分布参数
   - 形状参数 β = {beta:.2f} ({'磨损主导' if beta >1 else '随机失效'})
   - 尺度参数 η = {eta:.1f} 次循环 (特征寿命)

3. 工程建议
   - 建议维护周期: ≤ {b10_life} 次循环 (保证90%可靠性)
   - 重点关注 {survival_prob.index[np.argmax(failure_rates)]} 次循环附近的失效
   - 若需提高可靠性，建议优化胶层抗疲劳性 (β>1表明磨损是主因)
"""
print(report)