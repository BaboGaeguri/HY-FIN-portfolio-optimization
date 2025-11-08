import requests
import pandas as pd

def data():
    url = 'https://apis.data.go.kr/1160100/service/GetStockSecuritiesInfoService/getStockPriceInfo'

    params_base = {
        'serviceKey': 'yLTnNiCtiAKjrQ/q5g88bmjI60eII3JyvQdvSDbr7UqW/8LAmvPXxVWexltSH2oBU0vMjV64vJQ+9VRMi05EVg==',
        'numOfRows': '10000',
        'resultType': 'json',
        'beginBasDt': '20190101',
        'endBasDt': '20221231'
    }

    stocks = ['대한항공', '호텔신라', '강원랜드', 'CJ CGV', '롯데쇼핑']

    all_data = []

    for stock in stocks:
        params = params_base.copy()
        params['itmsNm'] = stock

        print(params)

        response = requests.get(url, params=params)

        # 1) HTTP 상태코드 체크
        if response.status_code != 200:
            print(f"[{stock}] HTTP {response.status_code}")
            print(response.text[:300])
            continue

        # 2) JSON 파싱 시도
        try:
            res_json = response.json()
        except ValueError:
            print(f"[{stock}] ❌ JSON이 아닌 응답")
            print(response.text[:300])
            continue

        # 3) 응답 구조에서 item 꺼내기
        body = res_json.get('response', {}).get('body', {})
        items = body.get('items', {}).get('item', [])

        if not items:
            print(f"[{stock}] 데이터 없음")
            continue
        
        all_data.extend(items)

    # 4) 전체 결과 DataFrame으로 변환
    if not all_data:
        raise ValueError("불러온 데이터가 없습니다. 위 로그를 확인하세요.")

    df = pd.DataFrame(all_data)

    # 5) 필요한 컬럼만 추리기
    df = df[['basDt', 'itmsNm', 'clpr']].copy()

    # 6) basDt를 날짜형으로 변환
    df['basDt'] = pd.to_datetime(df['basDt'])

    # 7) clpr을 숫자로 변환 (콤마 제거)
    df['clpr'] = (
        df['clpr']
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    )

    # 8) 종목 + 날짜 기준 정렬
    df = df.sort_values(by=['itmsNm', 'basDt'])

    # 9) 종목별 수익률 계산
    df['return'] = df.groupby('itmsNm')['clpr'].pct_change()

    return df