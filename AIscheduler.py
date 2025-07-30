# scheduler.py
import time
import subprocess
from datetime import datetime
import schedule

# 실행할 스크립트 경로
SCRIPT_PATH = "stock_db.py"
# 실행 시간 (24시간 형식)
SCHEDULE_TIME = "16:30"

def job():
    print(f"[{datetime.now()}] 스크립트 실행 시작")
    try:
        subprocess.run(["python", SCRIPT_PATH], check=True)
        print(f"[{datetime.now()}] ✅ 스크립트 실행 완료")
    except subprocess.CalledProcessError as e:
        print(f"[{datetime.now()}] ❌ 스크립트 실행 실패: {e}")

schedule.every().day.at(SCHEDULE_TIME).do(job)

print(f"스케줄러 시작, 매일 {SCHEDULE_TIME}에 '{SCRIPT_PATH}' 실행됩니다.")

while True:
    schedule.run_pending()
    time.sleep(60)
