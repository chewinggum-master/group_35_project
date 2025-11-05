import time
from datetime import datetime

OUTPUT_FILE = "tmux_test.txt"

def log_time(interval=5):
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while True:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{current_time}] Logging time...\n"
            print(log_line.strip())
            f.write(log_line)
            f.flush()  # 立即寫入檔案
            time.sleep(interval)

if __name__ == "__main__":
    log_time(5)
