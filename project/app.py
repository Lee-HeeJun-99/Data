import os
import glob
import asyncio
from collections import deque
from typing import List, Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from infer_runtime import GRURuntimeInferencer


# =========================
# 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_DIR = os.path.join(BASE_DIR, "dataset_out")
MODEL_PATH = os.path.join(BASE_DIR, "results", "single", "win_128", "gru_h64_l2", "best_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "prepared_dataset_multi", "win_128", "seq_scaler.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "static")


# =========================
# 모델/런타임 설정
# =========================
WINDOW = 128
STRIDE = 32
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.0
NUM_CLASSES = 4

DEFAULT_REPLAY_INTERVAL_SEC = 0.03
MAX_HISTORY = 500
ALERT_CONSECUTIVE_THRESHOLD = 5


# =========================
# FastAPI
# =========================
app = FastAPI(title="Industrial Sensor GRU Dashboard")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================
# 요청 모델
# =========================
class StartRequest(BaseModel):
    csv_file: str
    replay_interval: Optional[float] = None


# =========================
# 전역 상태
# =========================
state: Dict[str, Any] = {
    "running": False,
    "source_file": None,
    "latest_sensor": None,
    "latest_prediction": None,
    "history": deque(maxlen=MAX_HISTORY),
    "replay_interval": DEFAULT_REPLAY_INTERVAL_SEC,
    "current_row": 0,
    "total_rows": 0,
    "alert_active": False,
    "alert_count": 0,
    "system_status": "IDLE",   # IDLE / RUNNING / ALARM / STOPPED / ERROR
    "last_error": None,
}

clients: List[WebSocket] = []
replay_task: Optional[asyncio.Task] = None


# =========================
# 추론 엔진
# =========================
inferencer = GRURuntimeInferencer(
    model_path=MODEL_PATH,
    scaler_path=SCALER_PATH,
    window=WINDOW,
    stride=STRIDE,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    num_classes=NUM_CLASSES,
)


# =========================
# 유틸
# =========================
def get_csv_files():
    if not os.path.isdir(CSV_DIR):
        return []
    files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    return [os.path.basename(f) for f in files]


def clear_runtime_state():
    state["latest_sensor"] = None
    state["latest_prediction"] = None
    state["history"].clear()
    state["current_row"] = 0
    state["total_rows"] = 0
    state["alert_active"] = False
    state["alert_count"] = 0
    state["last_error"] = None


def update_alert_status(pred: Optional[Dict[str, Any]]):
    if pred is None:
        return

    pred_mode = pred["pred_mode"]

    if pred_mode != 0:
        state["alert_count"] += 1
    else:
        state["alert_count"] = 0

    state["alert_active"] = state["alert_count"] >= ALERT_CONSECUTIVE_THRESHOLD

    if state["alert_active"]:
        state["system_status"] = "ALARM"
    elif state["running"]:
        state["system_status"] = "RUNNING"


async def broadcast(payload: Dict[str, Any]):
    dead = []
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in clients:
            clients.remove(ws)


def make_status_payload() -> Dict[str, Any]:
    return {
        "type": "status",
        "running": state["running"],
        "source_file": state["source_file"],
        "latest_sensor": state["latest_sensor"],
        "latest_prediction": state["latest_prediction"],
        "history": list(state["history"]),
        "replay_interval": state["replay_interval"],
        "current_row": state["current_row"],
        "total_rows": state["total_rows"],
        "alert_active": state["alert_active"],
        "alert_count": state["alert_count"],
        "available_csv_files": get_csv_files(),
        "system_status": state["system_status"],
        "last_error": state["last_error"],
    }


def push_history_item(sensor: Dict[str, Any], pred: Optional[Dict[str, Any]]):
    history_item = {
        "row_index": sensor["row_index"],
        "I_meas": sensor["I_meas"],
        "HF_energy": sensor["HF_energy"],
        "HF_rms": sensor["HF_rms"],
        "T_meas": sensor["T_meas"],
        "true_mode": sensor["mode"],
        "pred_mode": pred["pred_mode"] if pred else None,
        "pred_mode_name": pred["pred_mode_name"] if pred else None,
        "confidence": pred["confidence"] if pred else None,
        "alert_active": state["alert_active"],
        "system_status": state["system_status"],
    }
    state["history"].append(history_item)
    return history_item


# =========================
# CSV 재생 루프
# =========================
async def replay_csv_loop(csv_path: str):
    try:
        print(f"[INFO] replay start: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        required_cols = ["I_meas", "HF_energy", "HF_rms", "T_meas"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")

        clear_runtime_state()
        state["source_file"] = os.path.basename(csv_path)
        state["running"] = True
        state["system_status"] = "RUNNING"
        state["total_rows"] = len(df)

        inferencer.reset()

        await broadcast(make_status_payload())

        for i, row in df.iterrows():
            if not state["running"]:
                print("[INFO] replay stop requested")
                break

            sensor = {
                "row_index": int(i),
                "I_meas": float(row["I_meas"]),
                "HF_energy": float(row["HF_energy"]),
                "HF_rms": float(row["HF_rms"]),
                "T_meas": float(row["T_meas"]),
                "mode": int(row["mode"]) if "mode" in df.columns else None,
            }

            pred = inferencer.step(sensor)
            update_alert_status(pred)

            state["latest_sensor"] = sensor
            state["latest_prediction"] = pred
            state["current_row"] = int(i + 1)

            history_item = push_history_item(sensor, pred)

            if i % 100 == 0:
                print(f"[DEBUG] row={i}, pred={pred}")

            await broadcast({
                "type": "update",
                "sensor": sensor,
                "prediction": pred,
                "history_item": history_item,
                "progress": {
                    "current_row": state["current_row"],
                    "total_rows": state["total_rows"],
                },
                "alert": {
                    "active": state["alert_active"],
                    "count": state["alert_count"],
                },
                "system_status": state["system_status"],
            })

            await asyncio.sleep(state["replay_interval"])

        if state["system_status"] != "ERROR":
            state["system_status"] = "STOPPED"

        print("[INFO] replay finished")

    except Exception as e:
        state["running"] = False
        state["system_status"] = "ERROR"
        state["last_error"] = str(e)
        print(f"[ERROR] replay failed: {repr(e)}")
        await broadcast({
            "type": "error",
            "message": str(e),
        })
    finally:
        state["running"] = False
        await broadcast(make_status_payload())


# =========================
# 라우트
# =========================
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/latest")
async def api_latest():
    return JSONResponse(make_status_payload())


@app.get("/api/history")
async def api_history():
    return JSONResponse({"history": list(state["history"])})


@app.get("/api/files")
async def api_files():
    return {"files": get_csv_files()}


@app.post("/api/start")
async def api_start(req: StartRequest):
    global replay_task

    print(f"[INFO] /api/start csv={req.csv_file}, interval={req.replay_interval}")

    if state["running"]:
        return {"ok": False, "message": "already running"}

    available = get_csv_files()
    if req.csv_file not in available:
        raise HTTPException(status_code=404, detail=f"CSV 파일을 찾을 수 없습니다: {req.csv_file}")

    state["replay_interval"] = (
        req.replay_interval
        if req.replay_interval is not None
        else DEFAULT_REPLAY_INTERVAL_SEC
    )

    csv_path = os.path.join(CSV_DIR, req.csv_file)
    replay_task = asyncio.create_task(replay_csv_loop(csv_path))

    return {
        "ok": True,
        "message": "replay started",
        "csv_file": req.csv_file,
        "replay_interval": state["replay_interval"],
    }


@app.post("/api/stop")
async def api_stop():
    state["running"] = False
    state["system_status"] = "STOPPED"
    return {"ok": True, "message": "replay stop requested"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)

    try:
        await websocket.send_json(make_status_payload())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)