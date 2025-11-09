import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 기본 계산 출력
# =========================================
def base_calculating(df):
    print("\n" + "=" * 80)
    print("종목별 평균 수익률과 표준편차")
    print("=" * 80)

    # Pivot 데이터프레임 생성
    pivot_df = df.pivot(index='date', columns='ticker', values='return')

    # 평균 수익률과 표준편차 계산
    mean_returns = pivot_df.mean()
    std_returns = pivot_df.std()

    # 결과 데이터프레임 생성
    stats_summary = pd.DataFrame({
        'Mean Return': mean_returns,
        'Std Dev': std_returns,
        'Annualized Mean (%)': mean_returns * 12 * 100,
        'Annualized Std (%)': std_returns * np.sqrt(12) * 100
    })

    print(stats_summary)

    print("\n" + "=" * 80)
    print("공분산 행렬 (Covariance Matrix)")
    print("=" * 80)

    covariance_matrix = pivot_df.cov()
    print(covariance_matrix)

    print("\n" + "=" * 80)
    print("상관계수 행렬 (Correlation Matrix)")
    print("=" * 80)

    correlation_matrix = pivot_df.corr()
    print(correlation_matrix)

    return stats_summary, covariance_matrix

# =========================================
# 상관계수 히트맵
# =========================================
def visualize_correlation_matrix(df):
    """
    Visualize correlation matrix as heatmap
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    """
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    correlation_matrix = pivot_df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_yticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticklabels(correlation_matrix.columns)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)
    
    # Add correlation values in cells
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('correlation_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nCorrelation heatmap saved as 'correlation_matrix_heatmap.png'")
    plt.show()

# =========================================
# 5가지 종목의 동일가중 포트폴리오
# =========================================
def equal_weight_portfolio(df):
    """
    Calculate equal weight portfolio statistics and perform backtesting
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    
    Returns:
    dict: Portfolio statistics including expected return, std dev, and cumulative return
    """
    # Pivot 데이터프레임 생성
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    
    # 동일가중 (각 종목 20%)
    n_assets = len(pivot_df.columns)
    weights = np.ones(n_assets) / n_assets
    
    # 포트폴리오 월별 수익률 계산
    portfolio_returns = (pivot_df * weights).sum(axis=1)
    
    # 포트폴리오 통계
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    portfolio_cumulative = (1 + portfolio_returns).prod() - 1
    
    # 연환산
    annualized_return = portfolio_mean * 12
    annualized_std = portfolio_std * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_std if annualized_std != 0 else 0
    
    print("\n" + "=" * 80)
    print("동일가중 포트폴리오 분석 (Equal Weight Portfolio)")
    print("=" * 80)
    print(f"\n가중치 (Weights):")
    for ticker, weight in zip(pivot_df.columns, weights):
        print(f"  {ticker}: {weight:.2%}")
    
    print(f"\n월별 평균 수익률: {portfolio_mean:.4%}")
    print(f"월별 표준편차: {portfolio_std:.4%}")
    print(f"\n연환산 기대수익률: {annualized_return:.2%}")
    print(f"연환산 표준편차: {annualized_std:.2%}")
    print(f"샤프 비율 (Sharpe Ratio): {sharpe_ratio:.4f}")
    print(f"\n12개월 누적수익률: {portfolio_cumulative:.2%}")
    
    # 백테스팅 결과 데이터프레임
    backtest_df = pd.DataFrame({
        'date': portfolio_returns.index,
        'monthly_return': portfolio_returns.values,
        'cumulative_return': (1 + portfolio_returns).cumprod() - 1
    })
    
    return {
        'weights': weights,
        'mean_return': portfolio_mean,
        'std_dev': portfolio_std,
        'annualized_return': annualized_return,
        'annualized_std': annualized_std,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': portfolio_cumulative,
        'backtest_df': backtest_df,
        'monthly_returns': portfolio_returns
    }

def equal_weight_portfolio_analysis(df):
    """
    Calculate expected return and standard deviation of equal-weighted portfolio
    and perform backtesting
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    
    Returns:
    dict: Portfolio statistics and backtest results
    """
    # Pivot 데이터프레임 생성
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    
    # 종목 수
    n_assets = len(pivot_df.columns)
    
    # 동일가중 (각 종목 20%)
    weights = np.array([1/n_assets] * n_assets)
    
    # 포트폴리오 월별 수익률 계산
    portfolio_returns = (pivot_df * weights).sum(axis=1)
    
    # 포트폴리오 통계
    portfolio_mean = portfolio_returns.mean()
    portfolio_std = portfolio_returns.std()
    portfolio_cumulative = (1 + portfolio_returns).cumprod() - 1
    
    # 연환산 통계
    annualized_return = portfolio_mean * 12
    annualized_std = portfolio_std * np.sqrt(12)
    sharpe_ratio = annualized_return / annualized_std if annualized_std != 0 else 0
    
    # 최종 누적수익률
    final_cumulative_return = portfolio_cumulative.iloc[-1]
    
    # 최대 낙폭 (Maximum Drawdown) 계산
    cumulative_wealth = (1 + portfolio_returns).cumprod()
    running_max = cumulative_wealth.cummax()
    drawdown = (cumulative_wealth - running_max) / running_max
    max_drawdown = drawdown.min()
    
    print("=" * 80)
    print("동일가중평균 포트폴리오 분석")
    print("=" * 80)
    print(f"\n포트폴리오 구성:")
    for ticker in pivot_df.columns:
        print(f"  {ticker}: {weights[list(pivot_df.columns).index(ticker)]:.2%}")
    
    print(f"\n월별 통계:")
    print(f"  평균 수익률: {portfolio_mean:.4%}")
    print(f"  표준편차: {portfolio_std:.4%}")
    
    print(f"\n연환산 통계:")
    print(f"  기대수익률: {annualized_return:.2%}")
    print(f"  표준편차 (변동성): {annualized_std:.2%}")
    print(f"  샤프 비율: {sharpe_ratio:.4f}")
    
    print(f"\n백테스팅 결과 (2024.01.31 ~ 2024.12.31):")
    print(f"  최종 누적수익률: {final_cumulative_return:.2%}")
    print(f"  최대 낙폭 (MDD): {max_drawdown:.2%}")
    
    results = {
        'weights': weights,
        'portfolio_returns': portfolio_returns,
        'portfolio_cumulative': portfolio_cumulative,
        'monthly_mean': portfolio_mean,
        'monthly_std': portfolio_std,
        'annualized_return': annualized_return,
        'annualized_std': annualized_std,
        'sharpe_ratio': sharpe_ratio,
        'final_cumulative_return': final_cumulative_return,
        'max_drawdown': max_drawdown
    }
    
    return results

def visualize_portfolio_backtest(df, portfolio_results):
    """
    Visualize equal-weighted portfolio backtest results
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    portfolio_results: Dictionary from equal_weight_portfolio_analysis
    """
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    
    # 개별 종목 누적수익률
    individual_cumulative = (1 + pivot_df).cumprod() - 1
    
    # 포트폴리오 누적수익률
    portfolio_cumulative = portfolio_results['portfolio_cumulative']
    
    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 1. 포트폴리오 vs 개별 종목 누적수익률
    colors = {'A': '#2E86AB', 'B': '#A23B72', 'C': '#F18F01', 'D': '#C73E1D', 'E': '#6A994E'}
    
    # 개별 종목 플롯
    for ticker in individual_cumulative.columns:
        ax1.plot(individual_cumulative.index, individual_cumulative[ticker] * 100,
                label=f'Ticker {ticker}', linewidth=1.5, color=colors.get(ticker), alpha=0.5, linestyle='--')
    
    # 포트폴리오 플롯 (굵게)
    ax1.plot(portfolio_cumulative.index, portfolio_cumulative * 100,
            label='Equal-Weight Portfolio', linewidth=3, color='black', marker='o', markersize=5)
    
    ax1.set_title('Portfolio vs Individual Assets - Cumulative Returns', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. 월별 수익률 바 차트
    portfolio_returns = portfolio_results['portfolio_returns']
    colors_bars = ['green' if r > 0 else 'red' for r in portfolio_returns]
    
    ax2.bar(portfolio_returns.index, portfolio_returns * 100, color=colors_bars, alpha=0.7, edgecolor='black')
    ax2.set_title('Equal-Weight Portfolio - Monthly Returns', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Monthly Return (%)', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('equal_weight_portfolio_backtest.png', dpi=300, bbox_inches='tight')
    print("\nPortfolio backtest chart saved as 'equal_weight_portfolio_backtest.png'")
    plt.show()

# =========================================
# 누적수익률 시각화
# =========================================
def visualize_cumulative_returns(df):
    """
    Visualize cumulative returns by ticker
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    """
    # Calculate cumulative returns for each ticker
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    cumulative_returns = (1 + pivot_df).cumprod() - 1
    
    # Set figure size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Cumulative Returns Line Chart
    colors = {'A': '#2E86AB', 'B': '#A23B72', 'C': '#F18F01', 'D': '#C73E1D', 'E': '#6A994E'}
    
    for ticker in cumulative_returns.columns:
        ax.plot(cumulative_returns.index, cumulative_returns[ticker] * 100, 
                label=f'Ticker {ticker}', linewidth=2.5, color=colors.get(ticker), marker='o', markersize=4)
    
    ax.set_title('Cumulative Returns by Ticker', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_cumulative_returns.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'portfolio_cumulative_returns.png'")
    plt.show()

# =========================================
# 월별수익률 시각화
# =========================================
def visualize_monthly_returns(df):
    """
    Visualize monthly returns by ticker
    
    Parameters:
    df: DataFrame with columns ['date', 'ticker', 'return']
    """
    # Pivot data for plotting
    pivot_df = df.pivot(index='date', columns='ticker', values='return')
    
    # Set figure size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Monthly Returns Line Chart
    colors = {'A': '#2E86AB', 'B': '#A23B72', 'C': '#F18F01', 'D': '#C73E1D', 'E': '#6A994E'}
    
    for ticker in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[ticker] * 100, 
                label=f'Ticker {ticker}', linewidth=2.5, color=colors.get(ticker), marker='o', markersize=4)
    
    ax.set_title('Monthly Returns by Ticker', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Monthly Return (%)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Rotate x-axis labels
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('portfolio_monthly_returns.png', dpi=300, bbox_inches='tight')
    print("\nChart saved as 'portfolio_monthly_returns.png'")
    plt.show()