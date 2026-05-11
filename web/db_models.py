from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    DateTime,
    Float,
    func,
)
from sqlalchemy.orm import relationship

from db import Base


class TimestampMixin:
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)

    permissions = relationship(
        "UserDevicePermission",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"User(id={self.id}, username='{self.username}', is_admin={self.is_admin})"


class Device(TimestampMixin, Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True, nullable=False)
    source_file = Column(String, nullable=False)
    display_name = Column(String, nullable=True)

    permissions = relationship(
        "UserDevicePermission",
        back_populates="device",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"Device(id={self.id}, device_id='{self.device_id}', "
            f"source_file='{self.source_file}', display_name='{self.display_name}')"
        )


class UserDevicePermission(TimestampMixin, Base):
    __tablename__ = "user_device_permissions"
    __table_args__ = (
        UniqueConstraint("user_id", "device_id", name="uq_user_device"),
    )

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)

    user = relationship("User", back_populates="permissions")
    device = relationship("Device", back_populates="permissions")

    def __repr__(self):
        return (
            f"UserDevicePermission(id={self.id}, user_id={self.user_id}, "
            f"device_id={self.device_id})"
        )


class SensorData(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True, nullable=False)

    row_index = Column(Integer, nullable=True)

    I_meas = Column(Float, nullable=False)
    HF_energy = Column(Float, nullable=False)
    HF_rms = Column(Float, nullable=False)
    hf_norm = Column(Float, nullable=False)
    arc_flag = Column(Float, nullable=False)
    arc_detect = Column(Float, nullable=False)
    T_meas = Column(Float, nullable=False)

    true_mode = Column(Integer, nullable=True)
    pred_mode = Column(Integer, nullable=True)
    pred_mode_name = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)

    alert_active = Column(Boolean, default=False, nullable=False)
    system_status = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def __repr__(self):
        return (
            f"SensorData(id={self.id}, device_id='{self.device_id}', "
            f"row_index={self.row_index}, I_meas={self.I_meas}, "
            f"HF_energy={self.HF_energy}, HF_rms={self.HF_rms}, "
            f"T_meas={self.T_meas})"
        )