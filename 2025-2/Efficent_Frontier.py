import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def simulate_random_portfolios(df, n_portfolios=10000, risk_free_rate=0.0):
    """
    Simulate random portfolios and visualize efficient frontier
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    n_portfolios: Number of random portfolios to generate
    risk_free_rate: Risk-free rate for Sharpe ratio calculation (annualized)
    
    Returns:
    dict: Portfolio simulation results
    """
    # Pivot 데이터프레임 생성
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    
    # 평균 수익률과 공분산 행렬
    mean_returns = pivot_df.mean()
    cov_matrix = pivot_df.cov()
    n_assets = len(pivot_df.columns)
    
    # 결과 저장용 배열
    results = np.zeros((4, n_portfolios))
    weights_record = []
    
    np.random.seed(42)
    
    for i in range(n_portfolios):
        # 랜덤 가중치 생성 (합이 1이 되도록)
        weights = np.random.random(n_assets)
        weights /= weights.sum()
        weights_record.append(weights)
        
        # 포트폴리오 수익률 (연환산)
        portfolio_return = np.sum(mean_returns * weights) * 12
        
        # 포트폴리오 표준편차 (연환산)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(12)
        
        # 샤프 비율
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        
        # 결과 저장
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = sharpe_ratio
        results[3, i] = portfolio_return / portfolio_std  # Return/Risk ratio
    
    # 최대 샤프 비율 포트폴리오
    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_return = results[0, max_sharpe_idx]
    max_sharpe_std = results[1, max_sharpe_idx]
    max_sharpe_weights = weights_record[max_sharpe_idx]
    
    # 최소 변동성 포트폴리오
    min_vol_idx = np.argmin(results[1])
    min_vol_return = results[0, min_vol_idx]
    min_vol_std = results[1, min_vol_idx]
    min_vol_weights = weights_record[min_vol_idx]
    
    print("\n" + "=" * 80)
    print(f"랜덤 포트폴리오 시뮬레이션 ({n_portfolios:,}개)")
    print("=" * 80)
    
    print(f"\n최대 샤프 비율 포트폴리오:")
    print(f"  연환산 수익률: {max_sharpe_return:.2%}")
    print(f"  연환산 표준편차: {max_sharpe_std:.2%}")
    print(f"  샤프 비율: {results[2, max_sharpe_idx]:.4f}")
    print(f"  가중치:")
    for ticker, weight in zip(pivot_df.columns, max_sharpe_weights):
        print(f"    {ticker}: {weight:.2%}")
    
    print(f"\n최소 변동성 포트폴리오:")
    print(f"  연환산 수익률: {min_vol_return:.2%}")
    print(f"  연환산 표준편차: {min_vol_std:.2%}")
    print(f"  샤프 비율: {results[2, min_vol_idx]:.4f}")
    print(f"  가중치:")
    for ticker, weight in zip(pivot_df.columns, min_vol_weights):
        print(f"    {ticker}: {weight:.2%}")
    
    return {
        'results': results,
        'weights_record': weights_record,
        'max_sharpe_idx': max_sharpe_idx,
        'min_vol_idx': min_vol_idx,
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix
    }

def visualize_simul_frontier(df, simulation_results, portfolio_stats):
    """
    Visualize efficient frontier with random portfolios
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    simulation_results: Dictionary returned from simulate_random_portfolios
    portfolio_stats: Dictionary returned from equal_weight_portfolio
    """
    results = simulation_results['results']
    weights_record = simulation_results['weights_record']
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 랜덤 포트폴리오 scatter plot
    ax.scatter(results[1, :] * 100, results[0, :] * 100, 
              marker='o', s=30, alpha=0.3, color='blue', edgecolors='black')
    
    # 여기부터 효율적 경계선 표시하는 코드-------
    # 효율적 경계선: 같은 표준편차에서 가장 높은 수익률을 가지는 포트폴리오를 선택
    efficient_indices = []
    for i in range(results.shape[1]):
        # 현재 포트폴리오보다 표준편차가 작거나 같은 다른 포트폴리오들 중
        # 더 높은 수익률이 있는지 확인
        if not np.any((results[1, :] <= results[1, i]) & (results[0, :] > results[0, i])):
            efficient_indices.append(i)

    # 효율적 경계선 포트폴리오 추출
    efficient_std = results[1, efficient_indices] * 100
    efficient_ret = results[0, efficient_indices] * 100

    # 효율적 경계선 점 (빨간색 별)
    ax.scatter(efficient_std, efficient_ret, 
               color='red', marker='*', s=100, 
               label='Efficient Frontier', zorder=10, edgecolors='black')
    # 여기까지-------
    '''
    # 각 포트폴리오의 가중치를 텍스트로 표시
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    tickers = pivot_df.columns
    
    for i in range(len(weights_record)):
        weights = weights_record[i]
        x = results[1, i] * 100
        y = results[0, i] * 100
        
        # 가중치를 문자열로 변환 (예: "20,30,15,20,15")
        weight_str = ','.join([f'{w*100:.0f}' for w in weights])
        
        ax.annotate(weight_str, 
                   xy=(x, y), 
                   xytext=(0, 0), 
                   textcoords='offset points',
                   fontsize=10,
                   alpha=0.6,
                   ha='center',
                   va='center')
        
    # 동일가중 포트폴리오 (녹색 다이아몬드)
    ax.scatter(portfolio_stats['annualized_std'] * 100, 
              portfolio_stats['annualized_return'] * 100,
              marker='D', color='green', s=30, edgecolors='black', linewidths=1,
              label='Equal Weight', zorder=5)
    '''
    ax.set_title('Efficient Frontier - Random Portfolio Simulation\n(Numbers show weights: A,B,C,D,E in %)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Annualized Standard Deviation (%)', fontsize=12)
    ax.set_ylabel('Annualized Expected Return (%)', fontsize=12)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=300, bbox_inches='tight')
    print("\nEfficient frontier chart saved as 'efficient_frontier.png'")
    plt.show()

