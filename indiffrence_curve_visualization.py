import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def plot_efficient_frontier_with_indifference_curves(df, 
                                                     simulation_results=None,
                                                     risk_aversion_levels=[2, 4, 6],
                                                     n_points_frontier=100):
    """
    효율적 경계선과 투자자의 무차별곡선을 함께 시각화
    위험회피계수에 따라 최적 포트폴리오가 어떻게 달라지는지 보여줌
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    simulation_results: 랜덤 시뮬레이션 결과 (선택사항)
    risk_aversion_levels: 위험회피계수 A 값들
    n_points_frontier: 효율적 경계선의 점 개수
    """
    # 효율적 경계선 계산
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    mean_returns = pivot_df.mean()
    cov_matrix = pivot_df.cov()
    n_assets = len(pivot_df.columns)
    
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    def portfolio_return(weights):
        return np.sum(mean_returns * weights) * 12
    
    def portfolio_std(weights):
        return np.sqrt(portfolio_variance(weights)) * np.sqrt(12)
    
    bounds = tuple((0, 1) for _ in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # 최소 분산 포트폴리오
    min_var_result = minimize(portfolio_variance, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)
    min_var_weights = min_var_result.x
    min_var_return = portfolio_return(min_var_weights)
    min_var_std = portfolio_std(min_var_weights)
    
    # 효율적 경계선 계산
    max_return = np.max(mean_returns) * 12
    target_returns = np.linspace(min_var_return, max_return * 0.99, n_points_frontier)
    efficient_stds = []
    efficient_returns = []
    
    for target_ret in target_returns:
        constraints_with_return = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_ret}
        )
        result = minimize(portfolio_variance, initial_weights,
                         method='SLSQP', bounds=bounds,
                         constraints=constraints_with_return,
                         options={'maxiter': 1000})
        if result.success:
            efficient_stds.append(portfolio_std(result.x))
            efficient_returns.append(target_ret)
    
    # 시각화
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 랜덤 포트폴리오 (있는 경우)
    if simulation_results is not None:
        results = simulation_results['results']
        ax.scatter(results[1, :] * 100, results[0, :] * 100, 
                  marker='o', s=20, alpha=0.2, color='lightgray', 
                  edgecolors='none', label='Random Portfolios')
    
    # 효율적 경계선
    ax.plot(np.array(efficient_stds) * 100, np.array(efficient_returns) * 100,
           'b-', linewidth=3, label='Efficient Frontier', zorder=4)
    
    # 각 위험회피계수에 대한 최적 포트폴리오와 무차별곡선
    colors = ['red', 'orange', 'purple']
    optimal_portfolios = []
    
    for i, A in enumerate(risk_aversion_levels):
        # 최적 포트폴리오 찾기: maximize U = μ - 0.5 * A * σ²
        def negative_utility(weights):
            ret = portfolio_return(weights)
            std = portfolio_std(weights)
            return -(ret - 0.5 * A * std**2)
        
        optimal_result = minimize(negative_utility, initial_weights,
                                 method='SLSQP', bounds=bounds, constraints=constraints)
        
        if optimal_result.success:
            opt_weights = optimal_result.x
            opt_return = portfolio_return(opt_weights)
            opt_std = portfolio_std(opt_weights)
            opt_utility = opt_return - 0.5 * A * opt_std**2
            
            optimal_portfolios.append({
                'A': A,
                'return': opt_return,
                'std': opt_std,
                'utility': opt_utility,
                'weights': opt_weights
            })
            
            # 최적 포트폴리오 점
            ax.scatter(opt_std * 100, opt_return * 100,
                      marker='*', s=500, color=colors[i], 
                      edgecolors='black', linewidths=2,
                      label=f'Optimal (A={A})', zorder=6)
            
            # 무차별곡선: U = μ - 0.5 * A * σ² = opt_utility
            # μ = opt_utility + 0.5 * A * σ²
            sigma_range = np.linspace(0, max(efficient_stds) * 1.5, 200)
            mu_indiff = opt_utility + 0.5 * A * sigma_range**2
            
            ax.plot(sigma_range * 100, mu_indiff * 100,
                   linestyle='--', linewidth=2, color=colors[i], alpha=0.7,
                   label=f'Indifference Curve (A={A})', zorder=3)
    
    ax.set_xlabel('Annualized Standard Deviation (%)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12)
    ax.set_title('Efficient Frontier with Indifference Curves\nOptimal Portfolio Selection by Risk Aversion', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # y축 범위 조정 (무차별곡선이 너무 위로 올라가지 않도록)
    if len(efficient_returns) > 0:
        ax.set_ylim(bottom=min(efficient_returns) * 100 * 0.8,
                   top=max(efficient_returns) * 100 * 1.3)
    
    plt.tight_layout()
    plt.savefig('efficient_frontier_with_indifference.png', dpi=300, bbox_inches='tight')
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("위험회피계수별 최적 포트폴리오")
    print("=" * 80)
    for opt in optimal_portfolios:
        print(f"\n위험회피계수 A = {opt['A']}:")
        print(f"  연환산 수익률: {opt['return']:.2%}")
        print(f"  연환산 표준편차: {opt['std']:.2%}")
        print(f"  효용: {opt['utility']:.4f}")
        print(f"  가중치: {', '.join([f'{t}={w:.1%}' for t, w in zip(pivot_df.columns, opt['weights'])])}")
    
    print("\nChart saved as 'efficient_frontier_with_indifference.png'")
    plt.show()