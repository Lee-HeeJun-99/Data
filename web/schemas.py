from typing import List, Optional
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserInfoResponse(BaseModel):
    username: str
    is_admin: bool


# =========================
# 관리자 기능용
# =========================
class AdminUserCreateRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False


class AdminPasswordResetRequest(BaseModel):
    new_password: str


class PermissionUpdateRequest(BaseModel):
    device_ids: List[str]


class AdminUserResponse(BaseModel):
    id: int
    username: str
    is_admin: bool


class AdminDeviceResponse(BaseModel):
    id: int
    device_id: str
    source_file: str
    display_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None