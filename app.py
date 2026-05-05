import os
import glob
import asyncio
from collections import deque
from typing import Dict, Any, Optional, List

import pandas as pd
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    status,
    Query,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from infer_runtime import HybridGRURuntimeInferencer
from db import get_db, SessionLocal
from db_models import User, Device, UserDevicePermission, SensorData
from auth import (
    create_access_token,
    authenticate_user,
    get_current_user,
    SECRET_KEY,
    ALGORITHM,
    hash_password,
)
from schemas import (
    LoginRequest,
    TokenResponse,
    UserInfoResponse,
    AdminUserCreateRequest,
    AdminPasswordResetRequest,
    PermissionUpdateRequest,
)


# =========================
# 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "dataset_out")
STATIC_DIR = os.path.join(BASE_DIR, "static")

MODEL_PATH = os.path.join(
    BASE_DIR, "results", "best_model_new_hybrid_64.pt"
)
SEQ_SCALER_PATH = os.path.join(
    BASE_DIR, "prepared_dataset_multi", "win_64", "seq_scaler.pkl"
)
FEAT_SCALER_PATH = os.path.join(
    BASE_DIR, "prepared_dataset_multi", "win_64", "feat_scaler.pkl"
)


# =========================
# 모델/런타임 설정
# =========================
WINDOW = 64
STRIDE = 16
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.0
NUM_CLASSES = 4  # 재학습 전까지 4클래스 유지

DEFAULT_REPLAY_INTERVAL = 0.03
MAX_HISTORY = 300
ALERT_THRESHOLD = 5
MINI_SERIES_LEN = 40
DB_COMMIT_BATCH = 50


# =========================
# 모델 로드 (지연 로딩 - 실패해도 서버 시작)
# =========================
def try_load_inferencer() -> Optional[HybridGRURuntimeInferencer]:
    try:
        inferencer = HybridGRURuntimeInferencer(
            model_path=MODEL_PATH,
            seq_scaler_path=SEQ_SCALER_PATH,
            feat_scaler_path=FEAT_SCALER_PATH,
            window=WINDOW,
            stride=STRIDE,
            seq_input_dim=4,
            feat_input_dim=11,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_classes=NUM_CLASSES,
        )
        print("[INFO] 모델 로드 성공")
        return inferencer
    except Exception as e:
        print(f"[WARN] 모델 로드 실패 - 추론 없이 실행됩니다: {e}")
        return None


# 서버 시작 시 한 번만 로드 시도
_global_inferencer: Optional[HybridGRURuntimeInferencer] = None


# =========================
# FastAPI
# =========================
app = FastAPI(title="Industrial Monitoring Service")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# =========================
# 요청 모델
# =========================
class StartRequest(BaseModel):
    replay_interval: Optional[float] = None
    mode: str = "csv"  # "csv" | "realtime"


class DeviceDisplayNameUpdateRequest(BaseModel):
    display_name: Optional[str] = None


class RealtimeSensorInput(BaseModel):
    I_meas: float
    HF_energy: float
    HF_rms: float
    T_meas: float
    HF_raw: Optional[float] = None
    mode: Optional[int] = None  # true label (있으면 함께 저장)


# =========================
# Device Runtime
# =========================
class DeviceRuntime:
    def __init__(
        self,
        device_id: str,
        csv_path: str,
        source_file: str,
        display_name: Optional[str] = None,
    ):
        self.device_id = device_id
        self.csv_path = csv_path
        self.source_file = source_file
        self.display_name = display_name or device_id

        self.running = False
        self.device_mode = "csv"        # "csv" | "realtime"
        self.latest_sensor = None
        self.latest_prediction = None
        self.history = deque(maxlen=MAX_HISTORY)
        self.current_row = 0
        self.total_rows = 0
        self.alert_active = False
        self.alert_count = 0
        self.system_status = "IDLE"
        self.last_error = None
        self.replay_interval = DEFAULT_REPLAY_INTERVAL
        self.task: Optional[asyncio.Task] = None
        self.clients: List[WebSocket] = []

        self.spark_i = deque(maxlen=MINI_SERIES_LEN)
        self.spark_hf_energy = deque(maxlen=MINI_SERIES_LEN)
        self.spark_hf_rms = deque(maxlen=MINI_SERIES_LEN)
        self.spark_t = deque(maxlen=MINI_SERIES_LEN)

    def reset(self):
        self.latest_sensor = None
        self.latest_prediction = None
        self.history.clear()
        self.current_row = 0
        self.total_rows = 0
        self.alert_active = False
        self.alert_count = 0
        self.system_status = "IDLE"
        self.last_error = None
        self.spark_i.clear()
        self.spark_hf_energy.clear()
        self.spark_hf_rms.clear()
        self.spark_t.clear()
        if _global_inferencer:
            _global_inferencer.reset()

    def _spark_dict(self):
        return {
            "I_meas":    list(self.spark_i),
            "HF_energy": list(self.spark_hf_energy),
            "HF_rms":    list(self.spark_hf_rms),
            "T_meas":    list(self.spark_t),
        }

    def summary(self):
        return {
            "device_id":         self.device_id,
            "display_name":      self.display_name,
            "source_file":       self.source_file,
            "running":           self.running,
            "device_mode":       self.device_mode,
            "system_status":     self.system_status,
            "alert_active":      self.alert_active,
            "alert_count":       self.alert_count,
            "current_row":       self.current_row,
            "total_rows":        self.total_rows,
            "latest_prediction": self.latest_prediction,
            "inferencer_ready":  _global_inferencer is not None,
            "sparks":            self._spark_dict(),
        }

    def detail(self):
        return {
            "device_id":         self.device_id,
            "display_name":      self.display_name,
            "source_file":       self.source_file,
            "running":           self.running,
            "device_mode":       self.device_mode,
            "system_status":     self.system_status,
            "alert_active":      self.alert_active,
            "alert_count":       self.alert_count,
            "current_row":       self.current_row,
            "total_rows":        self.total_rows,
            "latest_sensor":     self.latest_sensor,
            "latest_prediction": self.latest_prediction,
            "history":           list(self.history),
            "last_error":        self.last_error,
            "replay_interval":   self.replay_interval,
            "inferencer_ready":  _global_inferencer is not None,
            "sparks":            self._spark_dict(),
        }

    def push_sensor(self, sensor: Dict[str, Any]):
        self.spark_i.append(float(sensor["I_meas"]))
        self.spark_hf_energy.append(float(sensor["HF_energy"]))
        self.spark_hf_rms.append(float(sensor["HF_rms"]))
        self.spark_t.append(float(sensor["T_meas"]))


devices: Dict[str, DeviceRuntime] = {}


# =========================
# DB / 권한 유틸
# =========================
def get_allowed_device_ids(db: Session, user: User):
    if user.is_admin:
        return [d.device_id for d in db.query(Device).all()]
    rows = (
        db.query(Device.device_id)
        .join(UserDevicePermission, UserDevicePermission.device_id == Device.id)
        .filter(UserDevicePermission.user_id == user.id)
        .all()
    )
    return [r[0] for r in rows]


def ensure_device_access(device_id: str, db: Session, user: User):
    if user.is_admin:
        return True
    allowed = (
        db.query(UserDevicePermission)
        .join(Device, UserDevicePermission.device_id == Device.id)
        .filter(
            UserDevicePermission.user_id == user.id,
            Device.device_id == device_id
        )
        .first()
    )
    if not allowed:
        raise HTTPException(status_code=403, detail="해당 장비에 접근 권한이 없습니다.")
    return True


def ensure_admin(user: User):
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="관리자 권한이 필요합니다.")


def get_user_from_ws_token(token: str, db: Session) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="WebSocket 인증 실패",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def get_device_db_row(db: Session, device_id: str) -> Optional[Device]:
    return db.query(Device).filter(Device.device_id == device_id).first()


# =========================
# 장비 로드
# =========================
def load_devices_from_db(db: Session):
    devices.clear()
    rows = db.query(Device).order_by(Device.id.asc()).all()
    for row in rows:
        csv_path = os.path.join(CSV_DIR, row.source_file)
        if not os.path.exists(csv_path):
            continue
        devices[row.device_id] = DeviceRuntime(
            device_id=row.device_id,
            csv_path=csv_path,
            source_file=row.source_file,
            display_name=row.display_name or row.device_id,
        )


def sync_devices_with_csv_and_db(db: Session):
    csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
    existing_rows = db.query(Device).order_by(Device.id.asc()).all()
    existing_by_source = {d.source_file: d for d in existing_rows}
    next_index = len(existing_rows) + 1

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        if filename in existing_by_source:
            continue
        device_id = f"device_{next_index:03d}"
        next_index += 1
        new_device = Device(
            device_id=device_id,
            source_file=filename,
            display_name=device_id,
        )
        db.add(new_device)

    db.commit()
    load_devices_from_db(db)


# =========================
# 브로드캐스트 / 상태 업데이트
# =========================
async def broadcast_device(device: DeviceRuntime, payload: Dict[str, Any]):
    dead = []
    for ws in device.clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in device.clients:
            device.clients.remove(ws)


def update_alert(device: DeviceRuntime, pred: Optional[Dict[str, Any]]):
    if pred is None:
        return
    if pred["pred_mode"] != 0:
        device.alert_count += 1
    else:
        device.alert_count = 0
    device.alert_active = device.alert_count >= ALERT_THRESHOLD
    if device.alert_active:
        device.system_status = "ALARM"
    elif device.running:
        device.system_status = "RUNNING"


def run_inference(sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """모델이 있으면 추론, 없으면 None 반환"""
    if _global_inferencer is None:
        return None
    return _global_inferencer.step(sensor)


# =========================
# CSV 재생
# =========================
async def replay_device_csv(device: DeviceRuntime):
    db = SessionLocal()
    try:
        if not os.path.exists(device.csv_path):
            raise FileNotFoundError(f"CSV not found: {device.csv_path}")

        df = pd.read_csv(device.csv_path)

        required_cols = ["I_meas", "HF_energy", "HF_rms", "T_meas"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")

        device.reset()
        device.running = True
        device.device_mode = "csv"
        device.system_status = "RUNNING"
        device.total_rows = len(df)

        await broadcast_device(device, {"type": "status", **device.detail()})

        pending_db_rows = []

        for i, row in df.iterrows():
            if not device.running:
                break

            sensor = {
                "row_index": int(i),
                "I_meas":    float(row["I_meas"]),
                "HF_energy": float(row["HF_energy"]),
                "HF_rms":    float(row["HF_rms"]),
                "T_meas":    float(row["T_meas"]),
                "HF_raw":    float(row["HF_raw"]) if "HF_raw" in df.columns and pd.notna(row["HF_raw"]) else None,
                "mode":      int(row["mode"]) if "mode" in df.columns and pd.notna(row["mode"]) else None,
            }

            device.push_sensor(sensor)
            pred = run_inference(sensor)
            update_alert(device, pred)

            device.latest_sensor = sensor
            device.latest_prediction = pred
            device.current_row = int(i + 1)

            history_item = {
                "row_index":      sensor["row_index"],
                "true_mode":      sensor["mode"],
                "pred_mode_name": pred["pred_mode_name"] if pred else None,
                "confidence":     pred["confidence"] if pred else None,
                "alert_active":   device.alert_active,
                "system_status":  device.system_status,
            }
            device.history.append(history_item)

            pending_db_rows.append(
                SensorData(
                    device_id=device.device_id,
                    row_index=sensor["row_index"],
                    I_meas=sensor["I_meas"],
                    HF_raw=sensor["HF_raw"],
                    HF_energy=sensor["HF_energy"],
                    HF_rms=sensor["HF_rms"],
                    T_meas=sensor["T_meas"],
                    true_mode=sensor["mode"],
                    pred_mode=pred["pred_mode"] if pred else None,
                    pred_mode_name=pred["pred_mode_name"] if pred else None,
                    confidence=pred["confidence"] if pred else None,
                    alert_active=device.alert_active,
                    system_status=device.system_status,
                )
            )

            if len(pending_db_rows) >= DB_COMMIT_BATCH:
                db.add_all(pending_db_rows)
                db.commit()
                pending_db_rows.clear()

            await broadcast_device(device, {
                "type":          "update",
                "device_id":     device.device_id,
                "display_name":  device.display_name,
                "sensor":        sensor,
                "prediction":    pred,
                "history_item":  history_item,
                "current_row":   device.current_row,
                "total_rows":    device.total_rows,
                "alert_active":  device.alert_active,
                "alert_count":   device.alert_count,
                "system_status": device.system_status,
                "sparks":        device._spark_dict(),
            })

            await asyncio.sleep(device.replay_interval)

        if pending_db_rows:
            db.add_all(pending_db_rows)
            db.commit()
            pending_db_rows.clear()

        if device.system_status not in ("ERROR", "ALARM"):
            device.system_status = "STOPPED"

    except asyncio.CancelledError:
        device.running = False
        device.system_status = "STOPPED"
        try:
            db.rollback()
        except Exception:
            pass
        raise

    except Exception as e:
        device.running = False
        device.system_status = "ERROR"
        device.last_error = str(e)
        try:
            db.rollback()
        except Exception:
            pass
        try:
            await broadcast_device(device, {
                "type":      "error",
                "device_id": device.device_id,
                "message":   str(e),
            })
        except Exception:
            pass

    finally:
        device.running = False
        device.task = None
        db.close()
        try:
            await broadcast_device(device, {"type": "status", **device.detail()})
        except Exception:
            pass


# =========================
# Startup
# =========================
@app.on_event("startup")
async def startup_event():
    global _global_inferencer
    _global_inferencer = try_load_inferencer()

    db_gen = get_db()
    db = next(db_gen)
    try:
        sync_devices_with_csv_and_db(db)
    finally:
        db.close()


# =========================
# 정적 페이지
# =========================
@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


# =========================
# 인증 API
# =========================
@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    access_token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=access_token)


@app.get("/api/auth/me", response_model=UserInfoResponse)
async def me(current_user: User = Depends(get_current_user)):
    return UserInfoResponse(
        username=current_user.username,
        is_admin=current_user.is_admin
    )


# =========================
# 장비 API
# =========================
@app.get("/api/devices")
async def api_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    allowed_ids = set(get_allowed_device_ids(db, current_user))
    return {"devices": [d.summary() for d in devices.values() if d.device_id in allowed_ids]}


@app.get("/api/devices/{device_id}/latest")
async def api_device_latest(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_device_access(device_id, db, current_user)
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="device not found")
    return devices[device_id].detail()


@app.get("/api/devices/{device_id}/history")
async def api_device_history(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_device_access(device_id, db, current_user)
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="device not found")
    return {"history": list(devices[device_id].history)}


@app.post("/api/devices/{device_id}/start")
async def api_device_start(
    device_id: str,
    req: StartRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_device_access(device_id, db, current_user)
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="device not found")

    device = devices[device_id]

    if device.task and not device.task.done():
        return {"ok": False, "message": "already running"}

    if req.mode == "realtime":
        # 실시간 모드: 태스크 없이 running 상태만 켬
        device.reset()
        device.running = True
        device.device_mode = "realtime"
        device.system_status = "RUNNING"
        await broadcast_device(device, {"type": "status", **device.detail()})
        return {
            "ok":          True,
            "device_id":   device.device_id,
            "display_name":device.display_name,
            "mode":        "realtime",
        }
    else:
        # CSV 재생 모드
        replay_interval = req.replay_interval or DEFAULT_REPLAY_INTERVAL
        replay_interval = max(0.001, replay_interval)
        device.replay_interval = replay_interval
        device.device_mode = "csv"
        device.last_error = None
        device.task = asyncio.create_task(replay_device_csv(device))
        return {
            "ok":             True,
            "device_id":      device.device_id,
            "display_name":   device.display_name,
            "source_file":    device.source_file,
            "replay_interval":device.replay_interval,
            "mode":           "csv",
        }


@app.post("/api/devices/{device_id}/stop")
async def api_device_stop(
    device_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_device_access(device_id, db, current_user)
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="device not found")

    device = devices[device_id]
    device.running = False
    device.system_status = "STOPPED"

    if device.task and not device.task.done():
        device.task.cancel()

    await broadcast_device(device, {"type": "status", **device.detail()})
    return {"ok": True, "message": "stop requested"}


# =========================
# 실시간 센서 데이터 수신 API
# =========================
@app.post("/api/devices/{device_id}/data")
async def api_device_data(
    device_id: str,
    sensor_input: RealtimeSensorInput,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    실시간 모드에서 센서가 직접 데이터를 POST하는 엔드포인트.

    사용 예시:
    POST /api/devices/device_001/data
    Authorization: Bearer <token>
    {
        "I_meas": 12.5,
        "HF_energy": 0.003,
        "HF_rms": 0.045,
        "T_meas": 25.3,
        "HF_raw": 0.12
    }
    """
    ensure_device_access(device_id, db, current_user)
    if device_id not in devices:
        raise HTTPException(status_code=404, detail="device not found")

    device = devices[device_id]

    if not device.running or device.device_mode != "realtime":
        raise HTTPException(
            status_code=400,
            detail="장비가 실시간 모드로 실행 중이 아닙니다. /start (mode=realtime) 먼저 호출하세요."
        )

    sensor = {
        "row_index": device.current_row,
        "I_meas":    sensor_input.I_meas,
        "HF_energy": sensor_input.HF_energy,
        "HF_rms":    sensor_input.HF_rms,
        "T_meas":    sensor_input.T_meas,
        "HF_raw":    sensor_input.HF_raw,
        "mode":      sensor_input.mode,
    }

    device.push_sensor(sensor)
    pred = run_inference(sensor)
    update_alert(device, pred)

    device.latest_sensor = sensor
    device.latest_prediction = pred
    device.current_row += 1

    history_item = {
        "row_index":      sensor["row_index"],
        "true_mode":      sensor["mode"],
        "pred_mode_name": pred["pred_mode_name"] if pred else None,
        "confidence":     pred["confidence"] if pred else None,
        "alert_active":   device.alert_active,
        "system_status":  device.system_status,
    }
    device.history.append(history_item)

    # DB 저장
    try:
        db.add(SensorData(
            device_id=device_id,
            row_index=sensor["row_index"],
            I_meas=sensor["I_meas"],
            HF_raw=sensor["HF_raw"],
            HF_energy=sensor["HF_energy"],
            HF_rms=sensor["HF_rms"],
            T_meas=sensor["T_meas"],
            true_mode=sensor["mode"],
            pred_mode=pred["pred_mode"] if pred else None,
            pred_mode_name=pred["pred_mode_name"] if pred else None,
            confidence=pred["confidence"] if pred else None,
            alert_active=device.alert_active,
            system_status=device.system_status,
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[WARN] DB 저장 실패: {e}")

    # WebSocket 브로드캐스트
    await broadcast_device(device, {
        "type":          "update",
        "device_id":     device.device_id,
        "display_name":  device.display_name,
        "sensor":        sensor,
        "prediction":    pred,
        "history_item":  history_item,
        "current_row":   device.current_row,
        "total_rows":    device.total_rows,
        "alert_active":  device.alert_active,
        "alert_count":   device.alert_count,
        "system_status": device.system_status,
        "sparks":        device._spark_dict(),
    })

    return {
        "ok":            True,
        "row_index":     sensor["row_index"],
        "prediction":    pred,
        "alert_active":  device.alert_active,
        "system_status": device.system_status,
    }


# =========================
# 저장된 센서 로그 조회 API
# =========================
@app.get("/api/devices/{device_id}/logs")
async def api_device_logs(
    device_id: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_device_access(device_id, db, current_user)
    rows = (
        db.query(SensorData)
        .filter(SensorData.device_id == device_id)
        .order_by(SensorData.id.desc())
        .limit(limit)
        .all()
    )
    return {
        "device_id": device_id,
        "count": len(rows),
        "logs": [
            {
                "id":             r.id,
                "row_index":      r.row_index,
                "I_meas":         r.I_meas,
                "HF_raw":         r.HF_raw,
                "HF_energy":      r.HF_energy,
                "HF_rms":         r.HF_rms,
                "T_meas":         r.T_meas,
                "true_mode":      r.true_mode,
                "pred_mode":      r.pred_mode,
                "pred_mode_name": r.pred_mode_name,
                "confidence":     r.confidence,
                "alert_active":   r.alert_active,
                "system_status":  r.system_status,
                "created_at":     r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }


# =========================
# 관리자 API
# =========================
@app.get("/api/admin/users")
async def admin_list_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    users = db.query(User).order_by(User.id.asc()).all()
    return {
        "users": [
            {"id": u.id, "username": u.username, "is_admin": u.is_admin}
            for u in users
        ]
    }


@app.post("/api/admin/users")
async def admin_create_user(
    req: AdminUserCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="이미 존재하는 사용자명입니다.")
    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        is_admin=req.is_admin
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin}}


@app.post("/api/admin/users/{user_id}/reset-password")
async def admin_reset_password(
    user_id: int,
    req: AdminPasswordResetRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"ok": True, "message": "비밀번호가 초기화되었습니다."}


@app.get("/api/admin/devices")
async def admin_list_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    devices_db = db.query(Device).order_by(Device.id.asc()).all()
    return {
        "devices": [
            {
                "id":           d.id,
                "device_id":    d.device_id,
                "source_file":  d.source_file,
                "display_name": d.display_name,
                "created_at":   d.created_at.isoformat() if d.created_at else None,
                "updated_at":   d.updated_at.isoformat() if d.updated_at else None,
            }
            for d in devices_db
        ]
    }


@app.post("/api/admin/devices/reload")
async def admin_reload_devices(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    sync_devices_with_csv_and_db(db)
    return {
        "ok": True,
        "message": "CSV와 DB를 다시 동기화하고 장비 런타임을 새로 불러왔습니다.",
        "device_count": len(devices),
    }


@app.put("/api/admin/devices/{device_id}/display-name")
async def admin_update_device_display_name(
    device_id: str,
    req: DeviceDisplayNameUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    device_row = get_device_db_row(db, device_id)
    if not device_row:
        raise HTTPException(status_code=404, detail="장비를 찾을 수 없습니다.")
    new_name = (req.display_name or "").strip()
    device_row.display_name = new_name if new_name else device_row.device_id
    db.commit()
    db.refresh(device_row)
    if device_id in devices:
        devices[device_id].display_name = device_row.display_name
    return {
        "ok": True,
        "device": {
            "device_id":    device_row.device_id,
            "source_file":  device_row.source_file,
            "display_name": device_row.display_name,
            "created_at":   device_row.created_at.isoformat() if device_row.created_at else None,
            "updated_at":   device_row.updated_at.isoformat() if device_row.updated_at else None,
        }
    }


@app.get("/api/admin/permissions/{user_id}")
async def admin_get_user_permissions(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    rows = (
        db.query(Device.device_id)
        .join(UserDevicePermission, UserDevicePermission.device_id == Device.id)
        .filter(UserDevicePermission.user_id == user.id)
        .all()
    )
    return {
        "user": {"id": user.id, "username": user.username, "is_admin": user.is_admin},
        "device_ids": [r[0] for r in rows]
    }


@app.put("/api/admin/permissions/{user_id}")
async def admin_update_user_permissions(
    user_id: int,
    req: PermissionUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    ensure_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")

    target_devices = db.query(Device).filter(Device.device_id.in_(req.device_ids)).all()
    found_device_ids = {d.device_id for d in target_devices}
    missing = [d for d in req.device_ids if d not in found_device_ids]
    if missing:
        raise HTTPException(status_code=400, detail=f"존재하지 않는 device_id: {missing}")

    db.query(UserDevicePermission).filter(UserDevicePermission.user_id == user.id).delete()
    db.commit()

    for device in target_devices:
        db.add(UserDevicePermission(user_id=user.id, device_id=device.id))
    db.commit()

    return {"ok": True, "message": "권한이 저장되었습니다.", "user_id": user.id, "device_ids": req.device_ids}


# =========================
# WebSocket
# =========================
@app.websocket("/ws/{device_id}")
async def websocket_device(
    websocket: WebSocket,
    device_id: str,
    token: str = Query(...)
):
    db_gen = get_db()
    db = next(db_gen)

    try:
        user = get_user_from_ws_token(token, db)

        if not user.is_admin:
            allowed = (
                db.query(UserDevicePermission)
                .join(Device, UserDevicePermission.device_id == Device.id)
                .filter(
                    UserDevicePermission.user_id == user.id,
                    Device.device_id == device_id
                )
                .first()
            )
            if not allowed:
                await websocket.close(code=1008)
                return

        if device_id not in devices:
            await websocket.close(code=1008)
            return

        device = devices[device_id]
        await websocket.accept()
        device.clients.append(websocket)

        await websocket.send_json({"type": "status", **device.detail()})

        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await websocket.close(code=1008)
        except Exception:
            pass
    finally:
        if device_id in devices and websocket in devices[device_id].clients:
            devices[device_id].clients.remove(websocket)
        db.close()
