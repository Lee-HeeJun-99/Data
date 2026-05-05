import os
import re
import sys
import glob
from typing import Optional

from db import Base, engine, SessionLocal
from db_models import User, Device, UserDevicePermission
from auth import hash_password


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "dataset_out")


def create_user_if_not_exists(db, username: str, password: str, is_admin: bool = False):
    user = db.query(User).filter(User.username == username).first()
    if user:
        return user

    user = User(
        username=username,
        password_hash=hash_password(password),
        is_admin=is_admin,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def normalize_display_name_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    text = re.sub(r"[_\-]+", " ", stem)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "Unnamed Device"

    parts = text.split(" ")
    parts = [p.capitalize() if not p.isupper() else p for p in parts]
    return " ".join(parts)


def create_device_if_not_exists(
    db,
    device_id: str,
    source_file: str,
    display_name: Optional[str] = None
):
    device = db.query(Device).filter(Device.device_id == device_id).first()
    if device:
        updated = False

        if device.source_file != source_file:
            device.source_file = source_file
            updated = True

        if not device.display_name or device.display_name.strip() == "":
            device.display_name = display_name or device_id
            updated = True

        if updated:
            db.commit()
            db.refresh(device)

        return device

    device = Device(
        device_id=device_id,
        source_file=source_file,
        display_name=display_name or device_id,
        device_type="csv",
    )
    db.add(device)
    db.commit()
    db.refresh(device)
    return device


def add_permission_if_not_exists(db, user_id: int, device_id: int):
    exists = db.query(UserDevicePermission).filter(
        UserDevicePermission.user_id == user_id,
        UserDevicePermission.device_id == device_id
    ).first()

    if exists:
        return exists

    perm = UserDevicePermission(user_id=user_id, device_id=device_id)
    db.add(perm)
    db.commit()
    db.refresh(perm)
    return perm


def clear_permissions_for_user(db, user_id: int):
    db.query(UserDevicePermission).filter(
        UserDevicePermission.user_id == user_id
    ).delete()
    db.commit()


def init_database(reset_permissions: bool = False):
    """
    DB 초기화 함수

    Args:
        reset_permissions: True일 때만 user1/user2 권한을 초기화하고 재분배합니다.
                           기본값 False - 기존 권한을 유지합니다.
                           운영 중 실수로 권한이 날아가는 것을 방지합니다.
    """
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        admin = create_user_if_not_exists(db, "admin", "admin1234", is_admin=True)
        user1 = create_user_if_not_exists(db, "user1", "user1234", is_admin=False)
        user2 = create_user_if_not_exists(db, "user2", "user1234", is_admin=False)

        csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))

        if len(csv_files) == 0:
            print("[WARN] dataset_out 폴더에 CSV가 없습니다.")
            print(f"[WARN] 경로: {CSV_DIR}")

        device_objs = []
        for idx, csv_path in enumerate(csv_files, start=1):
            device_id = f"device_{idx:03d}"
            filename = os.path.basename(csv_path)
            display_name = normalize_display_name_from_filename(filename)

            device = create_device_if_not_exists(
                db=db,
                device_id=device_id,
                source_file=filename,
                display_name=display_name
            )
            device_objs.append(device)

        # --reset 플래그가 있을 때만 권한 초기화 및 재분배
        if reset_permissions:
            print("[INFO] --reset 플래그 감지: user1/user2 권한을 초기화하고 재분배합니다.")
            clear_permissions_for_user(db, user1.id)
            clear_permissions_for_user(db, user2.id)

            n = len(device_objs)
            mid = n // 2

            user1_devices = device_objs[:mid] if n > 0 else []
            user2_devices = device_objs[mid:] if n > 0 else []

            for device in user1_devices:
                add_permission_if_not_exists(db, user1.id, device.id)

            for device in user2_devices:
                add_permission_if_not_exists(db, user2.id, device.id)

        else:
            print("[INFO] 기존 권한을 유지합니다. 권한을 재분배하려면 --reset 옵션을 사용하세요.")

            # 기존 권한이 전혀 없는 경우에만 초기 분배 (최초 실행 시)
            user1_has_perms = db.query(UserDevicePermission).filter(
                UserDevicePermission.user_id == user1.id
            ).count()
            user2_has_perms = db.query(UserDevicePermission).filter(
                UserDevicePermission.user_id == user2.id
            ).count()

            if user1_has_perms == 0 and user2_has_perms == 0 and len(device_objs) > 0:
                print("[INFO] 권한이 없는 신규 사용자 감지: 초기 권한을 자동 분배합니다.")
                n = len(device_objs)
                mid = n // 2

                user1_devices = device_objs[:mid] if n > 0 else []
                user2_devices = device_objs[mid:] if n > 0 else []

                for device in user1_devices:
                    add_permission_if_not_exists(db, user1.id, device.id)

                for device in user2_devices:
                    add_permission_if_not_exists(db, user2.id, device.id)

        # 최종 권한 조회 (출력용)
        user1_devices = (
            db.query(Device)
            .join(UserDevicePermission, UserDevicePermission.device_id == Device.id)
            .filter(UserDevicePermission.user_id == user1.id)
            .all()
        )
        user2_devices = (
            db.query(Device)
            .join(UserDevicePermission, UserDevicePermission.device_id == Device.id)
            .filter(UserDevicePermission.user_id == user2.id)
            .all()
        )

        print("=" * 60)
        print("DB 초기화 완료")
        print("=" * 60)
        print("기본 계정:")
        print("  admin / admin1234")
        print("  user1 / user1234")
        print("  user2 / user1234")
        print()
        print(f"CSV 디렉터리: {CSV_DIR}")
        print(f"등록된 장비 수: {len(device_objs)}")
        print()

        if len(device_objs) > 0:
            print("장비 목록:")
            for d in device_objs:
                print(
                    f"  {d.device_id} | "
                    f"display_name={d.display_name} | "
                    f"source_file={d.source_file}"
                )

        print()
        print("권한 현황:")
        print(f"  user1 -> {len(user1_devices)}개 장비")
        print(f"  user2 -> {len(user2_devices)}개 장비")
        print("=" * 60)

    finally:
        db.close()


if __name__ == "__main__":
    # 사용법:
    #   python init_db.py          → 기존 권한 유지, 신규 장비/사용자만 추가
    #   python init_db.py --reset  → user1/user2 권한 초기화 후 재분배
    reset = "--reset" in sys.argv
    if reset:
        confirm = input("⚠️  권한을 초기화하면 기존 권한 설정이 모두 사라집니다. 계속하시겠습니까? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("취소되었습니다.")
            sys.exit(0)

    init_database(reset_permissions=reset)
