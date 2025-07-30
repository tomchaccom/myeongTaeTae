import pandas as pd
import yfinance as yf
import sqlite3
from datetime import datetime

# === 설정 ===
KOSPI_CSV = "kospi_names.csv"
KOSDAQ_CSV = "kosdaq_names.csv"
DB_PATH = "stock_history2.db"
START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# === 1. 종목 불러오기 ===
kospi = pd.read_csv(KOSPI_CSV)
kosdaq = pd.read_csv(KOSDAQ_CSV)

# 종목코드 6자리 패딩 후 시장 티커 붙이기
kospi["Code"] = kospi["Code"].astype(str).str.zfill(6) + ".KS"
kosdaq["Code"] = kosdaq["Code"].astype(str).str.zfill(6) + ".KQ"

all_stocks = pd.concat([kospi, kosdaq])
symbols = all_stocks['Code'].tolist()

# === 2. DB 연결 및 테이블 생성 ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    date TEXT,
    code TEXT,
    name TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    PRIMARY KEY (date, code)
);
""")

# === 3. 데이터 수집 및 저장 ===
for symbol in symbols:
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
        print(f"[DEBUG] {symbol} columns: {df.columns}")
        print(f"[DEBUG] {symbol} head:\n{df.head()}")

        if df.empty:
            print(f"⚠️ 데이터 없음: {symbol}")
            continue

        # 멀티 인덱스 컬럼 제거 (두 번째 레벨 제거)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        # 'Open' 컬럼 존재 여부 확인
        if 'Open' not in df.columns:
            print(f"⚠️ 'Open' 컬럼 없음: {symbol}")
            continue

        for index, row in df.iterrows():
            cursor.execute("""
                INSERT OR REPLACE INTO history (date, code, name, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                index.strftime("%Y-%m-%d"), symbol,
                all_stocks.loc[all_stocks['Code'] == symbol, 'Name'].values[0],
                float(row['Open']), float(row['High']),
                float(row['Low']), float(row['Close']),
                int(row['Volume'])
            ))
        print(f"✅ 저장 완료: {symbol}")
    except Exception as e:
        print(f"❌ 오류: {symbol} → {e}")

# === 4. 커밋 및 종료 ===
conn.commit()
conn.close()
print("🎉 3년치 주가 데이터 저장 완료")