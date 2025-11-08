import numpy as np
import matplotlib.pyplot as plt

def plot_efficient_frontier(mus, cov_matrix, n_points=200,
                            ax=None, color='blue', label='Efficient Frontier'):
    """
    Markowitz 이론에 기반한 연속적인 효율적 프론티어를 그리는 함수.

    Parameters
    ----------
    mus : 1D np.array
        각 자산의 기대수익률 벡터 (길이 N).
    cov_matrix : 2D np.array
        공분산 행렬 (N x N).
    n_points : int
        프론티어 위에서 찍을 점 개수.
    ax : matplotlib.axes.Axes or None
        이미 만들어진 Axes에 그리려면 넣고, 아니면 새로 만듦.
    color : str
        프론티어 선 색깔.
    label : str
        범례에 쓸 라벨.
    """
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

    # 타깃 기대수익률 범위
    target_returns = np.linspace(mus.min(), mus.max(), n_points)

    # 각 타깃 수익률에 대한 최소 분산 (프론티어)
    frontier_sigma = []
    for r in target_returns:
        var_p = (A * r**2 - 2 * B * r + C) / D   # σ_p^2(r)
        frontier_sigma.append(np.sqrt(var_p))

    frontier_sigma = np.array(frontier_sigma)

    # 플롯
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(frontier_sigma, target_returns,
            color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Risk (σ, Standard Deviation)", fontsize=12)
    ax.set_ylabel("Expected Return (μ)", fontsize=12)
    ax.set_title("Efficient Frontier", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    return frontier_sigma, target_returns, ax

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
# 3. 효율적 프론티어 플롯
# -------------------------------
frontier_sigma, frontier_mu, ax = plot_efficient_frontier(mus, cov_matrix)

# 개별 자산도 같이 찍고 싶으면:
# for name, info in assets.items():
#     ax.scatter(info['sigma'], info['mu'], marker='^', s=80, label=f"Asset {name}")
ax.legend()
plt.show()
