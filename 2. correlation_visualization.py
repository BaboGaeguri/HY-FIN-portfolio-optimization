import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 1. Asset parameters
# -------------------------------
mu1 = 0.08   # Expected return of asset 1
mu2 = 0.12   # Expected return of asset 2

sigma1 = 0.15  # Std dev of asset 1
sigma2 = 0.25  # Std dev of asset 2

rhos = [-0.5, 0.0, 0.5, 1.0]  # Correlation coefficients

# -------------------------------
# 2. Generate example datasets for each rho
# -------------------------------
np.random.seed(42)
num_obs = 300

example_datasets = {}

for rho in rhos:
    cov = [
        [sigma1**2, rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2]
    ]
    mean = [mu1 / 252, mu2 / 252]
    data = np.random.multivariate_normal(mean, cov, size=num_obs)
    df_example = pd.DataFrame(data, columns=['asset1', 'asset2'])
    corr_sample = df_example['asset1'].corr(df_example['asset2'])
    print(f"Target rho = {rho}, Sample correlation ≈ {corr_sample:.3f}")
    example_datasets[rho] = df_example

# -------------------------------
# 3. Calculate portfolio risk-return curve for each rho
# -------------------------------
weights = np.linspace(0, 1, 201)

# 상관계수 1인 애만 시각화
plt.figure(figsize=(10, 7))

portfolio_mu = []
portfolio_sigma = []

for w in weights:
    w1 = w
    w2 = 1 - w
    mu_p = w1 * mu1 + w2 * mu2
    var_p = (
        (w1**2) * (sigma1**2) +
        (w2**2) * (sigma2**2) +
        2 * w1 * w2 * rhos[3] * sigma1 * sigma2
    )
    sigma_p = np.sqrt(var_p)
    portfolio_mu.append(mu_p)
    portfolio_sigma.append(sigma_p)

plt.plot(portfolio_sigma, portfolio_mu, color='red', label=f"ρ = {rhos[3]}")

# Individual assets
plt.scatter(sigma1, mu1, color='black', marker='o', s=80, label='Asset 1')
plt.scatter(sigma2, mu2, color='gray',  marker='o', s=80, label='Asset 2')

# Add text labels showing each asset's parameters
plt.text(sigma1 + 0.005, mu1, f"Asset 1\nμ = {mu1:.2f}, σ = {sigma1:.2f}", fontsize=10, va='bottom')
plt.text(sigma2 + 0.005, mu2, f"Asset 2\nμ = {mu2:.2f}, σ = {sigma2:.2f}", fontsize=10, va='bottom')

# -------------------------------
# 4. Plot styling
# -------------------------------
plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
plt.ylabel("Expected Return", fontsize=12)
plt.title("Effect of Correlation on Portfolio Risk–Return Tradeoff", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle="--")
plt.legend(title="Correlation (ρ)")
plt.tight_layout()
plt.show()



# 전부 다 시각화
plt.figure(figsize=(10, 7))

for rho in rhos:
    portfolio_mu = []
    portfolio_sigma = []

    for w in weights:
        w1 = w
        w2 = 1 - w
        mu_p = w1 * mu1 + w2 * mu2
        var_p = (
            (w1**2) * (sigma1**2) +
            (w2**2) * (sigma2**2) +
            2 * w1 * w2 * rho * sigma1 * sigma2
        )
        sigma_p = np.sqrt(var_p)
        portfolio_mu.append(mu_p)
        portfolio_sigma.append(sigma_p)

    plt.plot(portfolio_sigma, portfolio_mu, label=f"ρ = {rho}")

# Individual assets
plt.scatter(sigma1, mu1, color='black', marker='o', s=80, label='Asset 1')
plt.scatter(sigma2, mu2, color='gray',  marker='o', s=80, label='Asset 2')

# Add text labels showing each asset's parameters
plt.text(sigma1 + 0.005, mu1, f"Asset 1\nμ = {mu1:.2f}, σ = {sigma1:.2f}", fontsize=10, va='bottom')
plt.text(sigma2 + 0.005, mu2, f"Asset 2\nμ = {mu2:.2f}, σ = {sigma2:.2f}", fontsize=10, va='bottom')

# -------------------------------
# 4. Plot styling
# -------------------------------
plt.xlabel("Risk (σ, Standard Deviation)", fontsize=12)
plt.ylabel("Expected Return", fontsize=12)
plt.title("Effect of Correlation on Portfolio Risk–Return Tradeoff", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle="--")
plt.legend(title="Correlation (ρ)")
plt.tight_layout()
plt.show()


