import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

# 스크립트 위치 기준으로 .env 로드 (실행 위치와 무관하게 동작)
load_dotenv(dotenv_path=Path(__file__).parent / '.env')

BASE_URL = 'https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo'

STOCKS = ['대한항공', '호텔신라', '강원랜드', 'CJ CGV', '롯데쇼핑']

BEGIN_DATE = '20230101'
END_DATE   = '20261231'


def fetch_stock(stock: str, service_key: str) -> list[dict]:
    params = {
        'serviceKey': service_key,
        'numOfRows': '10000',
        'resultType': 'json',
        'beginBasDt': BEGIN_DATE,
        'endBasDt': END_DATE,
        'itmsNm': stock,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)

    if response.status_code != 200:
        print(f"[{stock}] HTTP {response.status_code}")
        print(response.text[:300])
        return []

    try:
        res_json = response.json()
    except ValueError:
        print(f"[{stock}] JSON 파싱 실패")
        print(response.text[:300])
        return []

    body = res_json.get('response', {}).get('body', {})
    items = body.get('items', {}).get('item', [])

    if not items:
        print(f"[{stock}] 데이터 없음")
        return []

    return items if isinstance(items, list) else [items]


def data() -> pd.DataFrame:
    service_key = os.environ.get('DATA_SERVICE_KEY')
    if not service_key:
        raise EnvironmentError(
            "DATA_SERVICE_KEY 환경변수가 없습니다. "
            ".env 파일을 생성하거나 환경변수를 설정하세요."
        )

    all_items = []
    for stock in STOCKS:
        items = fetch_stock(stock, service_key)
        all_items.extend(items)

    if not all_items:
        raise ValueError("불러온 데이터가 없습니다. 위 로그를 확인하세요.")

    df = pd.DataFrame(all_items)

    df = df[['basDt', 'itmsNm', 'clpr']].copy()
    df['basDt'] = pd.to_datetime(df['basDt'], format='%Y%m%d')
    df['clpr'] = (
        df['clpr']
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    df = df.sort_values(by=['itmsNm', 'basDt']).reset_index(drop=True)
    df['return'] = df.groupby('itmsNm')['clpr'].pct_change()

    return df


if __name__ == '__main__':
    df = data()
    print(df.head(20))
    print(f"\n총 {len(df)}행, 종목: {df['itmsNm'].unique().tolist()}")
