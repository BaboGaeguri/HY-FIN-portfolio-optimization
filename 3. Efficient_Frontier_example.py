import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Define 5 assets (expected return, risk)
# -------------------------------
assets = {
    'A': {'mu': 0.065, 'sigma': 0.10},  # 안정형
    'B': {'mu': 0.085, 'sigma': 0.14},  # 중립형
    'C': {'mu': 0.100, 'sigma': 0.19},  # 성장형
    'D': {'mu': 0.125, 'sigma': 0.27},  # 공격형
    'E': {'mu': 0.145, 'sigma': 0.40}   # 초고위험/초고수익형
}

asset_names = list(assets.keys())
mus = np.array([a['mu'] for a in assets.values()])
sigmas = np.array([a['sigma'] for a in assets.values()])

# -------------------------------
# 2. Covariance matrix (assume moderate positive correlation)
# -------------------------------
rho = -0.25
cov_matrix = np.outer(sigmas, sigmas) * rho
np.fill_diagonal(cov_matrix, sigmas**2)

# -------------------------------
# 3. Generate random portfolios
# -------------------------------
np.random.seed(42)

num_portfolios = 100
port_mus = []
port_sigmas = []
port_weights = []

for _ in range(num_portfolios):
    w = np.random.random(len(assets))
    w /= np.sum(w)  # normalize weights to sum to 1
    port_weights.append(w)
    
    mu_p = np.dot(w, mus)
    sigma_p = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    
    port_mus.append(mu_p)
    port_sigmas.append(sigma_p)

port_mus = np.array(port_mus)
port_sigmas = np.array(port_sigmas)
port_weights = np.array(port_weights)

# -------------------------------
# 4. 랜덤 포트폴리오 시각화
# -------------------------------
plt.figure(figsize=(10, 7))

# Scatter 20 portfolio points
sc = plt.scatter(port_sigmas, port_mus, cmap='viridis', s=80, edgecolor='black', alpha=0.8)

# Annotate each point with weights (rounded to 1 decimal)
# for i in range(num_portfolios):
#     w_info = ", ".join([f"{name}:{port_weights[i][j]:.1f}" for j, name in enumerate(asset_names)])
#     plt.text(port_sigmas[i] + 0.002, port_mus[i], w_info, fontsize=8, va='center')

# Individual assets (reference points)
# for name, info in assets.items():
#     plt.scatter(info['sigma'], info['mu'], marker='^', s=100, label=f"Asset {name}")

plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
plt.ylabel("Expected Return (μ)", fontsize=12)
plt.title(f"Random Portfolios of 5 Assets ({num_portfolios} samples)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
# plt.colorbar(sc, label='Sharpe-like Ratio (μ/σ)')
plt.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------------
# 5. 효율적 프론티어 과정 시각화
# -------------------------------
# Sort portfolios by risk (σ)
sorted_indices = np.argsort(port_sigmas)
sorted_sigma = port_sigmas[sorted_indices]
sorted_mu = port_mus[sorted_indices]

# Keep only points that form the upper envelope (efficient frontier)
frontier_sigma = [sorted_sigma[0]]
frontier_mu = [sorted_mu[0]]

for i in range(1, len(sorted_mu)):
    if sorted_mu[i] > frontier_mu[-1]:
        frontier_sigma.append(sorted_sigma[i])
        frontier_mu.append(sorted_mu[i])

plt.figure(figsize=(10, 7))

# All portfolios (background)
plt.scatter(port_sigmas, port_mus, color='lightgray', alpha=0.5, label='Random Portfolios')

# Efficient frontier (highlighted)
plt.plot(frontier_sigma, frontier_mu, color='orange', linewidth=3, label='Efficient Frontier')
plt.scatter(frontier_sigma, frontier_mu, color='red', s=60, zorder=5)

# Individual assets
#for name, info in assets.items():
#     plt.scatter(info['sigma'], info['mu'], marker='^', s=100, label=f"Asset {name}")

# Styling
plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
plt.ylabel("Expected Return (μ)", fontsize=12)
plt.title("Efficient Frontier from Random Portfolios", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

# -------------------------------
# 6. 효율적 프론티어
# -------------------------------

Sigma = cov_matrix
Sigma_inv = np.linalg.inv(Sigma)
ones = np.ones(len(mus))

# Markowitz constants
A = ones @ Sigma_inv @ ones
B = ones @ Sigma_inv @ mus
C = mus @ Sigma_inv @ mus
D = A * C - B**2

# Target returns range (from min asset return to max)
target_returns = np.linspace(mus.min(), mus.max(), 200)

frontier_sigma_full = []
for r in target_returns:
    # Variance formula for efficient frontier:
    # σ_p^2(r) = (A r^2 - 2 B r + C) / D
    var_p = (A * r**2 - 2 * B * r + C) / D
    frontier_sigma_full.append(np.sqrt(var_p))

frontier_sigma_full = np.array(frontier_sigma_full)

plt.figure(figsize=(10, 7))

# Random portfolios (background)
plt.scatter(port_sigmas, port_mus, color='lightgray', alpha=0.5, label='Random Portfolios')

# Sample-based frontier (from step 5) – optional overlay
plt.plot(frontier_sigma, frontier_mu, color='orange', linewidth=2, label='Sample Frontier (Envelope)')
plt.scatter(frontier_sigma, frontier_mu, color='red', s=40, zorder=5)

# Analytical (continuous) efficient frontier
plt.plot(frontier_sigma_full, target_returns, color='blue', linewidth=3, label='Analytical Efficient Frontier')

plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
plt.ylabel("Expected Return (μ)", fontsize=12)
plt.title("Random Portfolios vs. Analytical Efficient Frontier", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()