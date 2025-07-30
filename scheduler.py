# scheduler.py
import time
from datetime import datetime
import schedule
from stock_db import save_today_all_stock_history

SCHEDULE_TIME = "20:00"
LOG_FILE = "scheduler.log"

def write_log(message: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {message}\n")

def job():
    msg_start = "스케줄러 작업 시작"
    print(f"[{datetime.now()}] {msg_start}")
    write_log(msg_start)
    try:
        save_today_all_stock_history()
        msg_success = "✅ 오늘 데이터 저장 완료"
        print(f"[{datetime.now()}] {msg_success}")
        write_log(msg_success)
    except Exception as e:
        msg_error = f"❌ 에러 발생: {e}"
        print(f"[{datetime.now()}] {msg_error}")
        write_log(msg_error)

schedule.every().day.at(SCHEDULE_TIME).do(job)

start_msg = f"스케줄러 시작, 매일 {SCHEDULE_TIME}에 'save_today_all_stock_history' 실행됩니다."
print(start_msg)
write_log(start_msg)

while True:
    schedule.run_pending()
    time.sleep(60)
