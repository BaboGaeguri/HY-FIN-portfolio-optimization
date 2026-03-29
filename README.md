# HY-FIN-portfolio-optimization
6주차 내용 관련

## 프로젝트 구조

```
HY-FIN-portfolio-optimization/
├── .venv/              # 가상환경 (git 제외)
├── .gitignore
├── requirements.txt
├── 2025-2/             # 2025년 2학기
│   ├── .env            # 서비스키 (git 제외)
│   ├── .env.example    # 서비스키 템플릿
│   └── return_data.py  # 주가 데이터 수집 (2019~2022)
└── 2026-1/             # 2026년 1학기
    ├── .env            # 서비스키 (git 제외)
    ├── .env.example    # 서비스키 템플릿
    └── return_data.py  # 주가 데이터 수집 (2024~2025)
```

## 환경 설정

### 1. 가상환경 활성화

**PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Git Bash:**
```bash
source .venv/Scripts/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
pip install python-dotenv
```

### 3. 서비스키 설정

[공공데이터포털](https://www.data.go.kr)에서 **주식시세정보 API** 서비스키를 발급받은 후,
각 학기 폴더에 `.env` 파일을 생성합니다.

```bash
# 2025-2 또는 2026-1 폴더 내에서
cp .env.example .env
```

`.env` 파일을 열고 키를 입력합니다:

```
DATA_SERVICE_KEY=발급받은_서비스키
```

## 실행

```bash
# 2026-1 데이터 수집
python 2026-1/return_data.py

# 2025-2 데이터 수집
python 2025-2/return_data.py
```
