import numpy as np
from scipy.optimize import minimize
import pandas as pd

def minimum_variance_portfolio(expected_returns, cov_matrix, asset_names):
    """
    최소분산 포트폴리오 (효율적 프론티어의 가장 왼쪽)
    
    Parameters:
    -----------
    expected_returns : np.array (n,) 또는 (n,1)
        각 자산의 기대수익률
    cov_matrix : np.array (n,n)
        자산간 공분산 행렬
    asset_names : list
        자산 이름 리스트
        
    Returns:
    --------
    pd.Series
        각 자산별 최적 비중
    """
    n = len(asset_names)
    
    # 목적함수: 포트폴리오 분산 최소화
    def objective(weights):
        return weights.T @ cov_matrix @ weights
    
    # 제약조건: 비중의 합 = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # 초기값
    initial_weights = np.ones(n) / n
    
    # 경계: 공매도 불가 (0 <= w <= 1)
    bounds = tuple((0, 1) for _ in range(n))
    
    # 최적화
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return pd.Series(result.x, index=asset_names, name='최소분산 포트폴리오')


def optimal_utility_portfolio(expected_returns, cov_matrix, asset_names, risk_aversion=1):
    """
    최적 포트폴리오 (효용함수 최대화)
    효용함수: U = w'μ - (λ/2) * w'Σw
    
    Parameters:
    -----------
    expected_returns : np.array (n,) 또는 (n,1)
        각 자산의 기대수익률
    cov_matrix : np.array (n,n)
        자산간 공분산 행렬
    asset_names : list
        자산 이름 리스트
    risk_aversion : float, default=1
        위험회피계수 (λ). 높을수록 위험을 더 회피
        
    Returns:
    --------
    pd.Series
        각 자산별 최적 비중
    """
    n = len(asset_names)
    expected_returns = np.array(expected_returns).flatten()
    
    # 목적함수: 효용함수 최대화 = -효용함수 최소화
    def objective(weights):
        portfolio_return = weights @ expected_returns
        portfolio_variance = weights.T @ cov_matrix @ weights
        utility = portfolio_return - (risk_aversion / 2) * portfolio_variance
        return -utility  # 최소화 문제로 변환
    
    # 제약조건: 비중의 합 = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # 초기값
    initial_weights = np.ones(n) / n
    
    # 경계: 공매도 불가 (0 <= w <= 1)
    bounds = tuple((0, 1) for _ in range(n))
    
    # 최적화
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return pd.Series(result.x, index=asset_names, name='최적 포트폴리오')


def tangency_portfolio(expected_returns, cov_matrix, asset_names, risk_free_rate=0):
    """
    탄젠시 포트폴리오 (샤프비율 최대화)
    
    Parameters:
    -----------
    expected_returns : np.array (n,) 또는 (n,1)
        각 자산의 기대수익률
    cov_matrix : np.array (n,n)
        자산간 공분산 행렬
    asset_names : list
        자산 이름 리스트
    risk_free_rate : float, default=0
        무위험이자율
        
    Returns:
    --------
    pd.Series
        각 자산별 최적 비중
    """
    n = len(asset_names)
    expected_returns = np.array(expected_returns).flatten()
    
    # 목적함수: 샤프비율 최대화 = -샤프비율 최소화
    def objective(weights):
        portfolio_return = weights @ expected_returns
        portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)
        
        # 분모가 0이 되는 것을 방지
        if portfolio_std < 1e-10:
            return 1e10
        
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe_ratio  # 최소화 문제로 변환
    
    # 제약조건: 비중의 합 = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # 초기값
    initial_weights = np.ones(n) / n
    
    # 경계: 공매도 불가 (0 <= w <= 1)
    bounds = tuple((0, 1) for _ in range(n))
    
    # 최적화
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return pd.Series(result.x, index=asset_names, name='탄젠시 포트폴리오')