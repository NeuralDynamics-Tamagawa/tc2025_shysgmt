import speedtest
import csv
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

# 測定間隔（秒）--- 5分 = 300秒
INTERVAL_SEC = 300

# 開始時刻をファイル名に使う
start_time = datetime.now()
filename_time = start_time.strftime("%Y-%m-%d_%H%M")
CSV_PATH = f"C:/Users/Sugimoto/internet_speed_log/internet_speed_log_{filename_time}.csv"

def measure_internet_speed():
    """インターネット速度を測定（下り／上り／Ping）"""
    st = speedtest.Speedtest()
    st.get_best_server()
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "download_mbps": round(st.download() / 1_000_000, 2),
        "upload_mbps": round(st.upload() / 1_000_000, 2),
        "ping_ms": round(st.results.ping, 2)
    }

def save_to_csv(record: dict, path: str):
    """測定結果をCSVに追記（初回はヘッダー付き）"""
    file_exists = Path(path).exists()
    with open(path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def wait_until_next_aligned_time(interval_sec: int):
    """次の「キリの良い時間（00分, 05分など）」まで待つ"""
    now = datetime.now()
    now_sec = now.minute * 60 + now.second
    next_aligned = math.ceil(now_sec / interval_sec) * interval_sec
    wait_sec = next_aligned - now_sec
    if wait_sec > 0:
        next_time = now + timedelta(seconds=wait_sec)
        print(f"⌛ 次の測定まで待機中: {next_time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(wait_sec)

def main():
    print(f"📡 インターネット速度を {INTERVAL_SEC // 60}分ごとに記録します")
    print(f"📄 ログファイル: {CSV_PATH}")
    
    # 最初のキリのよい時刻まで待機
    wait_until_next_aligned_time(INTERVAL_SEC)

    try:
        while True:
            start = time.time()

            try:
                result = measure_internet_speed()
                print(f"[{result['timestamp']}] ↓ {result['download_mbps']} Mbps | ↑ {result['upload_mbps']} Mbps | Ping {result['ping_ms']} ms")
                save_to_csv(result, CSV_PATH)
            except Exception as e:
                print(f"⚠️ 測定中にエラー: {e}")

            elapsed = time.time() - start
            time.sleep(max(0, INTERVAL_SEC - elapsed))

    except KeyboardInterrupt:
        print("\n✅ 測定を終了しました。")

if __name__ == "__main__":
    main()
