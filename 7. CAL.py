import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. 자산 A와 무위험자산 정의
# -------------------------------
r_f = 0.03  # risk-free rate
mu_A = 0.065  # expected return of risky asset A
sigma_A = 0.10  # std dev of risky asset A

# -------------------------------
# 2. CAL (Capital Allocation Line)
# -------------------------------
w = np.linspace(0, 2.0, 200)

expected_return = r_f + w * (mu_A - r_f)
portfolio_risk = w * sigma_A

# -------------------------------
# 3. 시각화
# -------------------------------
plt.figure(figsize=(9, 6))

# CAL 선
plt.plot(portfolio_risk, expected_return, color='orange', lw=2.5, label='CAL (Capital Allocation Line)')

# 위험자산 A 점 표시
plt.scatter(sigma_A, mu_A, color='red', s=100, marker='o', label='Risky Asset A')

# 무위험자산 점 표시
plt.scatter(0, r_f, color='blue', s=100, marker='o', label='Risk-free Asset')

# -------------------------------
# 4. 보조선 & 스타일
# -------------------------------
sharpe = (mu_A - r_f) / sigma_A
plt.text(sigma_A/2, r_f + sharpe * (sigma_A/2),
         f"Sharpe Ratio = {sharpe:.2f}", fontsize=10, color='gray')

plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
plt.ylabel("Expected Return (μ)", fontsize=12)
plt.title("Capital Allocation Line (Risk-free + Risky Asset A)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10)

# ✅ y축 0부터 시작
plt.ylim(bottom=0)

plt.tight_layout()
plt.show()
