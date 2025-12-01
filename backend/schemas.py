from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional, List

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Detection Schemas
class DetectionResult(BaseModel):
    status: str
    confidence: float
    timestamp: datetime
    eye_status: Optional[str] = None
    alert_triggered: bool = False
    
    class Config:
        from_attributes = True

class DetectionHistoryResponse(BaseModel):
    id: int
    session_id: str
    user_id: Optional[int]
    timestamp: datetime
    status: str
    confidence: float
    eye_status: Optional[str]
    duration_seconds: Optional[float]
    alert_triggered: bool
    
    class Config:
        from_attributes = True

# Session Schemas
class SessionCreate(BaseModel):
    session_id: str

class SessionResponse(BaseModel):
    id: int
    session_id: str
    user_id: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    total_alerts: int
    total_drowsy_time: float
    status: str
    
    class Config:
        from_attributes = True

class SessionStats(BaseModel):
    session_id: str
    user_id: Optional[int]
    start_time: datetime
    end_time: Optional[datetime]
    total_alerts: int
    total_drowsy_time: float
    total_detections: int
    drowsy_percentage: float
    status: str

# History Query
class HistoryQuery(BaseModel):
    session_id: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    skip: int = Field(default=0, ge=0)

class HistoryResponse(BaseModel):
    total: int
    items: List[DetectionHistoryResponse]