import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def calculate_portfolio_stats(df, portfolio_weights):
    """
    포트폴리오 통계 계산
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    portfolio_weights: dict 또는 Series, 각 자산의 비중
    
    Returns:
    dict: 포트폴리오 수익률, 표준편차, 가중치
    """
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    mean_returns = pivot_df.mean()
    cov_matrix = pivot_df.cov()
    
    # portfolio_weights를 numpy array로 변환
    if isinstance(portfolio_weights, dict):
        weights = np.array([portfolio_weights.get(ticker, 0) for ticker in pivot_df.columns])
    elif isinstance(portfolio_weights, pd.Series):
        weights = np.array([portfolio_weights.get(ticker, 0) for ticker in pivot_df.columns])
    else:
        weights = np.array(portfolio_weights)
    
    # 가중치 정규화
    weights = weights / weights.sum()
    
    # 통계 계산
    port_return = np.sum(mean_returns * weights) * 12
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_std = np.sqrt(port_variance) * np.sqrt(12)
    
    return {
        'return': port_return,
        'std': port_std,
        'weights': weights,
        'tickers': pivot_df.columns
    }

def plot_CAL_only(df, portfolio_weights, risk_free_rate=0.02):
    """
    포트폴리오와 무위험자산 간의 CAL만 시각화
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    portfolio_weights: dict 또는 Series, 각 자산의 비중
    risk_free_rate: float, 무위험수익률 (연환산)
    """
    # 포트폴리오 통계 계산
    stats = calculate_portfolio_stats(df, portfolio_weights)
    port_return = stats['return']
    port_std = stats['std']
    port_sharpe = (port_return - risk_free_rate) / port_std
    
    print("\n" + "=" * 80)
    print("포트폴리오 통계")
    print("=" * 80)
    print(f"\n포트폴리오 가중치:")
    for ticker, w in zip(stats['tickers'], stats['weights']):
        print(f"  {ticker}: {w:.2%}")
    print(f"\n연환산 기대수익률: {port_return:.2%}")
    print(f"연환산 표준편차: {port_std:.2%}")
    print(f"샤프 비율: {port_sharpe:.4f}")
    print(f"무위험수익률: {risk_free_rate:.2%}")
    
    # CAL 계산
    sigma_range = np.linspace(0, port_std * 1.5, 200)
    cal_returns = risk_free_rate + port_sharpe * sigma_range
    
    # 시각화
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # CAL (자본배분선)
    ax.plot(sigma_range * 100, cal_returns * 100,
           'r-', linewidth=3, label='CAL (Capital Allocation Line)', zorder=5)
    
    # 무위험자산
    ax.scatter(0, risk_free_rate * 100,
              marker='s', s=300, color='green', edgecolors='black', linewidths=2,
              label=f'Risk-Free Asset (Rf={risk_free_rate:.2%})', zorder=6)
    
    # 포트폴리오
    ax.scatter(port_std * 100, port_return * 100,
              marker='*', s=600, color='red', edgecolors='black', linewidths=2,
              label=f'Risky Portfolio (SR={port_sharpe:.2f})', zorder=6)
    
    # CAL 상의 예시 포인트들 (무위험자산과 포트폴리오의 선형결합)
    for y in [0.25, 0.5, 0.75]:
        combined_std = y * port_std
        combined_return = (1 - y) * risk_free_rate + y * port_return
        ax.scatter(combined_std * 100, combined_return * 100,
                  marker='o', s=150, color='orange', edgecolors='black', linewidths=2, 
                  zorder=5, alpha=0.7)
        # 레이블 추가
        ax.annotate(f'{y:.0%} risky', 
                   xy=(combined_std * 100, combined_return * 100),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    ax.set_xlabel('Annualized Standard Deviation (%)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12)
    ax.set_title('Capital Allocation Line (CAL)\nCombining Risk-Free Asset and Risky Portfolio', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=-1)
    
    plt.tight_layout()
    plt.savefig('cal_only.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'cal_only.png'")
    plt.show()

def plot_CAL_with_frontier(df, portfolio_weights, simulation_results, 
                           risk_free_rate):
    """
    효율적 프론티어와 CAL을 함께 시각화
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    portfolio_weights: dict 또는 Series, 각 자산의 비중
    simulation_results: Dictionary from simulate_random_portfolios
    risk_free_rate: float, 무위험수익률 (연환산)
    """
    # 포트폴리오 통계 계산
    stats = calculate_portfolio_stats(df, portfolio_weights)
    port_return = stats['return']
    port_std = stats['std']
    port_sharpe = (port_return - risk_free_rate) / port_std
    
    print("\n" + "=" * 80)
    print("포트폴리오 통계")
    print("=" * 80)
    print(f"\n포트폴리오 가중치:")
    for ticker, w in zip(stats['tickers'], stats['weights']):
        print(f"  {ticker}: {w:.2%}")
    print(f"\n연환산 기대수익률: {port_return:.2%}")
    print(f"연환산 표준편차: {port_std:.2%}")
    print(f"샤프 비율: {port_sharpe:.4f}")
    print(f"무위험수익률: {risk_free_rate:.2%}")
    
    # 시뮬레이션 결과
    results = simulation_results['results']
    
    # 효율적 경계선 포트폴리오 찾기
    efficient_indices = []
    for i in range(results.shape[1]):
        if not np.any((results[1, :] <= results[1, i]) & (results[0, :] > results[0, i])):
            efficient_indices.append(i)
    
    efficient_std = results[1, efficient_indices] * 100
    efficient_ret = results[0, efficient_indices] * 100
    
    # 효율적 경계선 정렬 (왼쪽에서 오른쪽으로)
    sort_idx = np.argsort(efficient_std)
    efficient_std = efficient_std[sort_idx]
    efficient_ret = efficient_ret[sort_idx]
    
    # CAL 계산
    sigma_range = np.linspace(0, max(results[1, :]) * 1.2, 200)
    cal_returns = risk_free_rate + port_sharpe * sigma_range
    
    # 시각화
    fig, ax = plt.subplots(figsize=(14, 10))

    # 랜덤 포트폴리오
    ax.scatter(results[1, :] * 100, results[0, :] * 100, 
              marker='o', s=20, alpha=0.2, color='lightgray', 
              edgecolors='none', label='Random Portfolios')
    
    # 효율적 경계선 (선으로 연결)
    ax.plot(efficient_std, efficient_ret, 
           'b-', linewidth=2.5, label='Efficient Frontier', zorder=4)
    ax.scatter(efficient_std, efficient_ret, 
               color='blue', marker='o', s=50, 
               zorder=5, edgecolors='darkblue')
    
    # CAL (자본배분선)
    ax.plot(sigma_range * 100, cal_returns * 100,
           'r--', linewidth=2.5, label='CAL (Capital Allocation Line)', zorder=6)
    
    # 무위험자산
    ax.scatter(0, risk_free_rate * 100,
              marker='s', s=300, color='green', edgecolors='black', linewidths=2,
              label=f'Risk-Free Asset (Rf={risk_free_rate:.1%})', zorder=7)
    
    # 포트폴리오
    ax.scatter(port_std * 100, port_return * 100,
              marker='*', s=600, color='red', edgecolors='black', linewidths=2,
              label=f'Portfolio (SR={port_sharpe:.2f})', zorder=7)
    
    # 개별 자산들
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    mean_returns_annual = pivot_df.mean() * 12
    std_returns_annual = pivot_df.std() * np.sqrt(12)
    
    '''
    # 개별자산들 플롯
    for ticker in pivot_df.columns:
        ax.scatter(std_returns_annual[ticker] * 100, mean_returns_annual[ticker] * 100,
                  marker='o', s=150, edgecolors='black', linewidths=2,
                  label=f'Asset {ticker}', zorder=5, alpha=0.7)
    '''
    ax.set_xlabel('Annualized Standard Deviation (%)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12)
    ax.set_title('Efficient Frontier with Capital Allocation Line (CAL)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=-1)
    
    plt.tight_layout()
    plt.savefig('cal_with_frontier.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'cal_with_frontier.png'")
    plt.show()

def plot_CAL_frontier_indifference(df, portfolio_weights, simulation_results, 
                                   risk_free_rate=0.02, risk_aversion_levels=[2, 4, 6]):
    """
    효율적 프론티어, CAL, 무차별곡선을 함께 시각화
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    portfolio_weights: dict 또는 Series, 각 자산의 비중
    simulation_results: Dictionary from simulate_random_portfolios
    risk_free_rate: float, 무위험수익률 (연환산)
    risk_aversion_levels: list, 위험회피계수 A 값들
    """
    # 포트폴리오 통계 계산
    stats = calculate_portfolio_stats(df, portfolio_weights)
    port_return = stats['return']
    port_std = stats['std']
    port_sharpe = (port_return - risk_free_rate) / port_std
    
    print("\n" + "=" * 80)
    print("포트폴리오 통계")
    print("=" * 80)
    print(f"\n포트폴리오 가중치:")
    for ticker, w in zip(stats['tickers'], stats['weights']):
        print(f"  {ticker}: {w:.2%}")
    print(f"\n연환산 기대수익률: {port_return:.2%}")
    print(f"연환산 표준편차: {port_std:.2%}")
    print(f"샤프 비율: {port_sharpe:.4f}")
    print(f"무위험수익률: {risk_free_rate:.2%}")
    
    # 시뮬레이션 결과
    results = simulation_results['results']
    
    # 효율적 경계선 포트폴리오 찾기
    efficient_indices = []
    for i in range(results.shape[1]):
        if not np.any((results[1, :] <= results[1, i]) & (results[0, :] > results[0, i])):
            efficient_indices.append(i)
    
    efficient_std = results[1, efficient_indices] * 100
    efficient_ret = results[0, efficient_indices] * 100
    
    # 효율적 경계선 정렬
    sort_idx = np.argsort(efficient_std)
    efficient_std = efficient_std[sort_idx]
    efficient_ret = efficient_ret[sort_idx]
    
    # CAL 계산
    sigma_range = np.linspace(0, max(results[1, :]) * 1.2, 200)
    cal_returns = risk_free_rate + port_sharpe * sigma_range
    
    # 각 위험회피계수에 대한 최적 포트폴리오 계산 (CAL 상에서)
    # CAL 상의 최적: maximize U = μ - 0.5 * A * σ²
    # CAL: μ = Rf + SR * σ
    # U = Rf + SR * σ - 0.5 * A * σ²
    # dU/dσ = SR - A * σ = 0
    # σ* = SR / A
    colors = ['purple', 'orange', 'brown']
    optimal_portfolios = []
    
    for A in risk_aversion_levels:
        # CAL 상의 최적 표준편차
        optimal_sigma = port_sharpe / A
        optimal_return = risk_free_rate + port_sharpe * optimal_sigma
        optimal_utility = optimal_return - 0.5 * A * optimal_sigma**2
        
        # 위험자산 비중 계산: y = σ* / σp
        y = optimal_sigma / port_std
        
        optimal_portfolios.append({
            'A': A,
            'sigma': optimal_sigma,
            'return': optimal_return,
            'utility': optimal_utility,
            'y': y
        })
    
    # 시각화
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 랜덤 포트폴리오
    ax.scatter(results[1, :] * 100, results[0, :] * 100, 
              marker='o', s=20, alpha=0.2, color='lightgray', 
              edgecolors='none', label='Random Portfolios')
    
    # 효율적 경계선
    ax.plot(efficient_std, efficient_ret, 
           'b-', linewidth=2.5, label='Efficient Frontier', zorder=4)
    ax.scatter(efficient_std, efficient_ret, 
               color='blue', marker='o', s=50, 
               zorder=5, edgecolors='darkblue')
    
    # CAL (자본배분선)
    ax.plot(sigma_range * 100, cal_returns * 100,
           'r--', linewidth=2.5, label='CAL (Capital Allocation Line)', zorder=6)
    
    # 무위험자산
    ax.scatter(0, risk_free_rate * 100,
              marker='s', s=300, color='green', edgecolors='black', linewidths=2,
              label=f'Risk-Free Asset (Rf={risk_free_rate:.1%})', zorder=7)
    
    # 위험 포트폴리오
    ax.scatter(port_std * 100, port_return * 100,
              marker='*', s=600, color='red', edgecolors='black', linewidths=2,
              label=f'Risky Portfolio (SR={port_sharpe:.2f})', zorder=7)
    
    # 각 위험회피계수에 대한 최적점과 무차별곡선
    for i, (opt, color) in enumerate(zip(optimal_portfolios, colors)):
        A = opt['A']
        sigma_opt = opt['sigma']
        return_opt = opt['return']
        utility_opt = opt['utility']
        
        # 최적 포트폴리오 점
        ax.scatter(sigma_opt * 100, return_opt * 100,
                  marker='D', s=300, color=color, 
                  edgecolors='black', linewidths=2,
                  label=f'Optimal (A={A}, y={opt["y"]:.1%})', zorder=8)
        
        # 무차별곡선: U = μ - 0.5 * A * σ² = utility_opt
        # μ = utility_opt + 0.5 * A * σ²
        sigma_indiff = np.linspace(0, max(sigma_range) * 1.1, 200)
        mu_indiff = utility_opt + 0.5 * A * sigma_indiff**2
        
        ax.plot(sigma_indiff * 100, mu_indiff * 100,
               linestyle=':', linewidth=2, color=color, alpha=0.7,
               label=f'Indifference (A={A})', zorder=3)
    
    ax.set_xlabel('Annualized Standard Deviation (%)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12)
    ax.set_title('CAL, Efficient Frontier and Indifference Curves\nOptimal Portfolio Selection with Risk-Free Asset', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=-1)
    
    # y축 범위 조정
    max_return = max(efficient_ret.max(), cal_returns.max() * 100)
    ax.set_ylim(bottom=min(efficient_ret.min(), risk_free_rate * 100) * 0.9,
               top=max_return * 1.2)
    
    plt.tight_layout()
    plt.savefig('cal_frontier_indifference.png', dpi=300, bbox_inches='tight')
    
    # 최적 포트폴리오 정보 출력
    print("\n" + "=" * 80)
    print("위험회피계수별 최적 포트폴리오 (CAL 상)")
    print("=" * 80)
    for opt in optimal_portfolios:
        print(f"\n위험회피계수 A = {opt['A']}:")
        print(f"  최적 표준편차: {opt['sigma']:.2%}")
        print(f"  최적 기대수익률: {opt['return']:.2%}")
        print(f"  위험자산 비중 (y): {opt['y']:.2%}")
        print(f"  무위험자산 비중: {1-opt['y']:.2%}")
        print(f"  효용: {opt['utility']:.4f}")
    
    print("\nChart saved as 'cal_frontier_indifference.png'")
    plt.show()