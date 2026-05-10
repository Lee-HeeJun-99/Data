import torch
import socket
import time
import pandas as pd
import random
import os
import glob

# --- 📂 1. 절대 경로 설정 ---
BASE_DIR = r"C:\Users\Administrator\Downloads"
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# --- 📡 2. UDP 통신 설정 ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
NUM_ACUS = 8  # 🌟 실외기 8대 설정

print("AI 모델 로딩 중...")

# --- 📊 4. CSV 데이터 로드 ---
modes = [0, 1, 2, 3]
data_frames = {}

print("CSV 파일 로딩 중...")
for m in modes:
    pattern = os.path.join(BASE_DIR, f"mode{m}*.csv")
    matched_files = glob.glob(pattern)
    
    if matched_files:
        df_list = [pd.read_csv(f) for f in matched_files]
        data_frames[m] = pd.concat(df_list, ignore_index=True)
        print(f"✅ mode{m} 데이터 로드 완료! (총 {len(matched_files)}개 병합)")
    else:
        print(f"❌ 경고: {pattern} 파일이 없습니다.")

def get_random_data():
    selected_mode = random.choice(list(data_frames.keys()))
    df = data_frames[selected_mode]
    random_row = df.sample(n=1).iloc[0]
    
    temp = random_row.get('HF_rms', 25.0)  
    arc = random_row.get('T_meas', 0.0)    
    
    return selected_mode, temp, arc

print("\n🔥 실시간 AI 추론 및 유니티 스트리밍 시작 (1초 간격)")

try:
    while True:
        if not data_frames:
            break

        # 🌟 0번부터 7번까지 8대 연속 발사!
        for acu_id in range(NUM_ACUS):
            actual_mode, temp, arc = get_random_data()
            
            # 아크 값이 너무 작아서 100만 곱하기 적용!
            arc_display = arc * 1000000 

            message = f"{acu_id},{actual_mode},{temp:.1f},{arc_display:.1f}"
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))
            
            print(f"전송 🚀 [{acu_id}번 실외기] 모드: {actual_mode}, 온도: {temp:.1f}, 아크: {arc_display:.1f}")
        
        # 🌟 이 줄 긋기가 나와야 진짜 새 코드입니다!
        print("-" * 40) 
        time.sleep(1.0) 

except KeyboardInterrupt:
    print("\n스트리밍을 종료합니다.")
    sock.close()