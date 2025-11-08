from return_data import data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 폰트 설정 ======
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['font.family'] = 'Malgun Gothic'   # Windows
plt.rcParams['axes.unicode_minus'] = False

# ====== 1. 데이터 불러오기 ======
df = data()   # basDt, itmsNm, clpr, return 포함됨

# ====== 2. 결측치 제거 ======
df = df.dropna(subset=['return'])
df = df.loc[df['itmsNm'] != '대한항공']
df = df.loc[df['basDt'] >= '2021-06-01']

# ====== 3. 일별 평균 수익률 계산 ======
daily_avg_return = (
    df.groupby('basDt')['return'].mean()
)  # 하루에 여러 종목의 평균 수익률

daily_avg_return_1 = (
    df.groupby('itmsNm')['return'].mean()
)
# ====== 4. 누적 수익률 계산 ======
initial_value = 100
cumulative_value = (1 + daily_avg_return).cumprod() * initial_value

# ====== 5. 개별 종목 누적 수익률도 같이 보기 ======
cumulative_by_stock = (
    df.pivot(index='basDt', columns='itmsNm', values='return')
       .apply(lambda x: (1 + x).cumprod() * initial_value)
)

# ====== 6. 시각화 ======
fig, ax = plt.subplots(figsize=(12, 7))

# 개별 종목 (얇은 선)
for stock in cumulative_by_stock.columns:
    ax.plot(
        cumulative_by_stock.index,
        cumulative_by_stock[stock],
        linewidth=1.2,
        alpha=0.8,
        label=stock
    )

# 동일가중 포트폴리오 (굵은 선)
ax.plot(
    cumulative_value.index,
    cumulative_value.values,
    color='#2563eb',
    linewidth=3,
    label='동일가중 포트폴리오 평균'
)

ax.set_title('일별 누적 수익률 추이 (초기 투자금: 100)', fontsize=14, fontweight='bold')
ax.set_xlabel('날짜', fontsize=12)
ax.set_ylabel('투자 가치', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

plt.tight_layout()
plt.show()

# ====== 7. 요약 출력 ======
print("=" * 60)
print("분석 결과 요약")
print("=" * 60)
print(f"기간: {df['basDt'].min().date()} ~ {df['basDt'].max().date()}")
print(f"총 거래일 수: {daily_avg_return.shape[0]}일")

print("\n[종목별 평균 수익률]")
print(daily_avg_return_1)

final_value = cumulative_value.iloc[-1]
total_return = (final_value - initial_value)
print(f"\n최종 누적 가치: {final_value:.2f}")
print(f"총 수익률: {total_return:.2f}%")
print("=" * 60)
