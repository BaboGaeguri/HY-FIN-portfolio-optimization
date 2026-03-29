import numpy as np
import matplotlib.pyplot as plt

def plot_efficient_frontier(mus, cov_matrix, n_points=200,
                            ax=None, color='blue', label='Efficient Frontier'):
    mus = np.asarray(mus)
    Sigma = np.asarray(cov_matrix)

    Sigma_inv = np.linalg.inv(Sigma)
    ones = np.ones(len(mus))

    # Markowitz constants
    A = ones @ Sigma_inv @ ones
    B = ones @ Sigma_inv @ mus
    C = mus  @ Sigma_inv @ mus
    D = A * C - B**2

    if D <= 0:
        raise ValueError("D <= 0: 공분산/기대수익 조합이 이상해서 프론티어를 계산할 수 없음.")

    target_returns = np.linspace(mus.min(), mus.max(), n_points)

    frontier_sigma = []
    for r in target_returns:
        var_p = (A * r**2 - 2 * B * r + C) / D   # σ_p^2(r)
        frontier_sigma.append(np.sqrt(var_p))

    frontier_sigma = np.array(frontier_sigma)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(frontier_sigma, target_returns,
            color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Risk (σ, Standard Deviation)", fontsize=12)
    ax.set_ylabel("Expected Return (μ)", fontsize=12)
    ax.set_title("Efficient Frontier with Indifference Curves", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    return frontier_sigma, target_returns, ax


def find_optimal_on_frontier(frontier_sigma, frontier_mu, A):
    """주어진 위험회피계수 A에서 프론티어 위 효용 최대 포인트 찾기"""
    utility = frontier_mu - 0.5 * A * frontier_sigma**2
    idx = np.argmax(utility)
    return frontier_sigma[idx], frontier_mu[idx], utility[idx]


# -------------------------------
# 1. 자산 정의
# -------------------------------
assets = {
    'A': {'mu': 0.065, 'sigma': 0.10},
    'B': {'mu': 0.085, 'sigma': 0.14},
    'C': {'mu': 0.100, 'sigma': 0.19},
    'D': {'mu': 0.125, 'sigma': 0.27},
    'E': {'mu': 0.145, 'sigma': 0.40}
}

asset_names = list(assets.keys())
mus = np.array([a['mu'] for a in assets.values()])
sigmas = np.array([a['sigma'] for a in assets.values()])

# -------------------------------
# 2. 공분산 행렬 (예: 상관계수 0.2 가정)
# -------------------------------
rho = 0.2
cov_matrix = np.outer(sigmas, sigmas) * rho
np.fill_diagonal(cov_matrix, sigmas**2)

# -------------------------------
# 3. 효율적 프론티어 계산 및 기본 플롯
# -------------------------------
frontier_sigma, frontier_mu, ax = plot_efficient_frontier(mus, cov_matrix)

# 개별 자산도 표시하고 싶으면 주석 해제
# for name, info in assets.items():
#     ax.scatter(info['sigma'], info['mu'], marker='^', s=70, label=f"Asset {name}")

# -------------------------------
# 4. 위험회피계수별 무차별곡선 + 접점 표시
# -------------------------------
risk_aversion_levels = [0.5, 3, 9]  # A 값들
sigma_range = np.linspace(0, frontier_sigma.max()*1.1, 300)

for A in risk_aversion_levels:
    # 프론티어 위 최적 포인트(접점)
    opt_sigma, opt_mu, U_opt = find_optimal_on_frontier(frontier_sigma, frontier_mu, A)

    # 이 효용 수준을 기준으로 한 무차별곡선
    mu_curve = U_opt + 0.5 * A * sigma_range**2

    ax.plot(sigma_range, mu_curve,
            linestyle='--', linewidth=1.8,
            label=f'Indifference (A={A})')

    # 접점 표시
    ax.scatter(opt_sigma, opt_mu,
               s=80, edgecolor='black', linewidth=1.0,
               label=f'Optimal (A={A})')

# -------------------------------
# 5. 마무리
# -------------------------------
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
