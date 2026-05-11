import torch
import paho.mqtt.client as mqtt  # 🌟 UDP 소켓 대신 MQTT 라이브러리 사용!
import json                      # 🌟 데이터를 JSON으로 묶기 위해 추가
import time
import pandas as pd
import random
import os
import glob

# --- 📂 1. 절대 경로 설정 ---
BASE_DIR = r"C:\Users\Administrator\Downloads"
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# --- 📡 2. MQTT 통신 설정 (UDP에서 변경됨) ---
BROKER_IP = "127.0.0.1"  # PC에서 직접 돌리면 127.0.0.1, 라즈베리파이면 PC의 IP 주소 입력
MQTT_TOPIC = "acu/data"

# MQTT 클라이언트 생성 및 브로커 연결
client = mqtt.Client("ACU_Data_Generator")
client.connect(BROKER_IP)
client.loop_start() # 백그라운드 통신 유지

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

print("\n🔥 실시간 AI 추론 및 유니티 스트리밍 시작 (MQTT 통신, 1초 간격)")

try:
    while True:
        if not data_frames:
            break

        # 🌟 0번부터 7번까지 8대 연속 발행!
        for acu_id in range(NUM_ACUS):
            actual_mode, temp, arc = get_random_data()
            
            # 아크 값이 너무 작아서 100만 곱하기 적용!
            arc_display = arc * 1000000 

            # 🌟 데이터를 JSON(딕셔너리) 형태로 예쁘게 포장 (유니티 코드랑 깔맞춤)
            payload = {
                "id": acu_id,
                "mode": int(actual_mode),
                "temp": float(temp),
                "arc": float(arc_display)
            }

            # 포장된 데이터를 문자열로 변환
            message = json.dumps(payload)
            
            # 브로커의 "acu/data" 토픽으로 메시지 쏘기!
            client.publish(MQTT_TOPIC, message)
            
            print(f"발행 🚀 [{acu_id}번 실외기] 모드: {actual_mode}, 온도: {temp:.1f}, 아크: {arc_display:.1f}")
        
        print("-" * 40) 
        time.sleep(1.0) 

except KeyboardInterrupt:
    print("\n스트리밍을 종료합니다.")
    client.loop_stop()
    client.disconnect()