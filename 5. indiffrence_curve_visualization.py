import numpy as np
import matplotlib.pyplot as plt

def plot_indifference_curves(risk_aversion_levels=[2, 4, 6],
                             base_utility=0.08,
                             sigma_max=0.4,
                             n_points=200):
    """
    투자자의 효용함수 U = μ - ½Aσ² 에 따른 무차별곡선을 시각화.

    Parameters
    ----------
    risk_aversion_levels : list of float
        위험회피계수 A 값들. (클수록 위험을 더 싫어함)
    base_utility : float
        기준 효용 수준 (곡선의 시작점 높이 조정용)
    sigma_max : float
        x축(σ)의 최대 범위
    n_points : int
        곡선을 그릴 구간 세분화 정도
    """
    sigma = np.linspace(0, sigma_max, n_points)
    plt.figure(figsize=(10, 7))

    for A in risk_aversion_levels:
        mu = base_utility + 0.5 * A * sigma**2
        plt.plot(sigma, mu, linestyle='--', linewidth=2, label=f'A = {A}')

    plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
    plt.ylabel("Expected Return (μ)", fontsize=12)
    plt.title("Investor Indifference Curves", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(title="Risk Aversion (A)", fontsize=10)
    plt.tight_layout()
    plt.show()

# 실행 예시
plot_indifference_curves(risk_aversion_levels=[0.5, 3, 9], base_utility=0.07)
