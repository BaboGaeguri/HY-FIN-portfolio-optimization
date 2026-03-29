import pandas as pd
import numpy as np

# 날짜 생성 (2024년 1월 31일 ~ 12월 31일, 월말 기준)
dates = pd.date_range(start='2024-01-31', end='2024-12-31', freq='ME')
n_months = len(dates)

# 시드 설정 (재현성을 위해, 원하면 제거 가능)
np.random.seed(42)

# 각 종목별 설정
# target_cumulative: 누적수익률 (12개월 후)
# volatility: 월별 수익률의 표준편차 (변동성)
assets_config = {
    'A': {'target_cumulative': 0.05, 'volatility': 0.02},   # 5%, 변동성 보통
    'B': {'target_cumulative': 0.03, 'volatility': 0.035},  # 3%, 변동성 조금 높음
    'C': {'target_cumulative': 0.10, 'volatility': 0.06},   # 10%, 변동성 아주 높음
    'D': {'target_cumulative': -0.02, 'volatility': 0.008}, # -2%, 변동성 아주 낮음
    'E': {'target_cumulative': -0.04, 'volatility': 0.022}  # -4%, 변동성 보통
}

# 데이터 생성
data = []

for ticker, config in assets_config.items():
    target_cum = config['target_cumulative']
    volatility = config['volatility']
    
    # 목표 누적수익률을 달성하기 위한 월별 평균 수익률
    # 단순 산술평균 방식: 월평균 * 12 = 연간 누적
    monthly_mean = target_cum / n_months
    
    # 정규분포를 따르는 월별 수익률 생성
    monthly_returns = np.random.normal(monthly_mean, volatility, n_months)
    
    # 실제 합계가 목표 누적수익률과 일치하도록 조정
    current_sum = monthly_returns.sum()
    adjustment = (target_cum - current_sum) / n_months
    adjusted_returns = monthly_returns + adjustment
    
    # 데이터 추가
    for date, ret in zip(dates, adjusted_returns):
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'ticker': ticker,
            'return': ret
        })

# 데이터프레임 생성
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# 날짜와 티커로 정렬
df = df.sort_values(['date', 'ticker']).reset_index(drop=True)

# 데이터 확인
print("=" * 80)
print("생성된 데이터 샘플 (처음 20개)")
print("=" * 80)
print(df.head(20))

print("\n" + "=" * 80)
print("데이터 정보")
print("=" * 80)
print(f"총 데이터 포인트: {len(df)}")
print(f"날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
print(f"종목 수: {df['ticker'].nunique()}")

# 각 종목별 통계 확인
print("\n" + "=" * 80)
print("종목별 통계")
print("=" * 80)

for ticker in sorted(df['ticker'].unique()):
    ticker_data = df[df['ticker'] == ticker]['return']
    
    # 누적수익률 계산 (단순 합산)
    cumulative_return = ticker_data.sum()
    
    # 월별 평균 수익률
    mean_return = ticker_data.mean()
    
    # 월별 표준편차 (변동성)
    std_return = ticker_data.std()
    
    print(f"\n종목 {ticker}:")
    print(f"  누적수익률: {cumulative_return:>8.2%} (목표: {assets_config[ticker]['target_cumulative']:>6.2%})")
    print(f"  월평균 수익률: {mean_return:>8.4%}")
    print(f"  월별 변동성: {std_return:>8.4%} (목표: {assets_config[ticker]['volatility']:>6.4%})")
    print(f"  최대 수익률: {ticker_data.max():>8.4%}")
    print(f"  최소 수익률: {ticker_data.min():>8.4%}")

print("\n" + "=" * 80)

# CSV로 저장 (선택사항)
df.to_csv('example_returns.csv', index=False)
print("\n데이터가 'example_returns.csv'로 저장되었습니다.")