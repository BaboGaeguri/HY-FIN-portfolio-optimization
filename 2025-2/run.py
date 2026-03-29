import pandas as pd
import numpy as np

from return_visualization import base_calculating, visualize_correlation_matrix
from return_visualization import visualize_monthly_returns, visualize_cumulative_returns
from return_visualization import equal_weight_portfolio, equal_weight_portfolio_analysis, visualize_portfolio_backtest
from Efficent_Frontier import simulate_random_portfolios, visualize_simul_frontier
from indiffrence_curve_visualization import plot_efficient_frontier_with_indifference_curves
from optimizer import minimum_variance_portfolio, optimal_utility_portfolio, tangency_portfolio
from CAL import plot_CAL_only, plot_CAL_with_frontier, plot_CAL_frontier_indifference

df = pd.read_csv('example_returns.csv')

portfolio_stats = equal_weight_portfolio(df)

# 효율적 프론티어 시각화
a= simulate_random_portfolios(df, 100000)
# visualize_simul_frontier(df, a, portfolio_stats)
# b = plot_efficient_frontier_with_indifference_curves(df, simulation_results=None, risk_aversion_levels=[0.5, 3, 9], n_points_frontier=100)

expected_return, cov_matrix = base_calculating(df)

# A와 CAL
# only_A = np.array([1,0,0,0,0])
# only_A_weight = pd.Series(only_A, index=['A', 'B', 'C', 'D', 'E'], name='A 종목')
# c = plot_CAL_only(df, only_A, risk_free_rate=0.002)

# 탄젠시 최적화
d = tangency_portfolio(expected_return['Mean Return'], cov_matrix, cov_matrix.columns, risk_free_rate=0.002)
# print('탄젠시 포트폴리오 최적화 비중')
# for ticker, w in d.items():
#     print(f"{ticker:<5}: {w*100:>6.2f}%")

# 탄젠시와 CAL 시각화
# 3 = plot_CAL_with_frontier(df, d, a, risk_free_rate=0.002)

# 탄젠시+CAL+무차별곡선
f = plot_CAL_frontier_indifference(df, d, a, 
                               risk_free_rate=0.002, risk_aversion_levels=[20, 30, 40])

'''
# 행렬식 변환 예시(자료 20p.)
a, cov_matrix = base_calculating(df)
mean_return = np.array([[0.004167], [0.002500], [0.008333], [-0.001667], [-0.003333]])
print('mean_return')
print(mean_return)

weight = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
print('weight')
print(weight)

expected_return = weight.T @ mean_return
print('expected_return')
print(expected_return)

weight = np.array([[0.2], [0.2], [0.2], [0.2], [0.2]])
print('weight')
print(weight)

pivot_df = df.pivot(index='date', columns='ticker', values='return')
cov_matrix = pivot_df.cov()
print('cov_matrix')
print(cov_matrix)

portfolio_risk = weight.T @ cov_matrix.values @ weight
print('portfolio_risk')
print(portfolio_risk)
'''