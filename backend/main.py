from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy import or_
from datetime import datetime
from typing import List
import cv2
import numpy as np
import base64
import json
import logging
import uuid
import os
from jose import JWTError, jwt

from database import get_db, init_db, DetectionHistory, SessionInfo, User
from model_loader import DrowsinessDetector
from schemas import (
    DetectionResult, DetectionHistoryResponse,
    SessionResponse, SessionStats, HistoryResponse,
    UserCreate, UserResponse, Token
)
from security import (
    verify_password,
    get_password_hash,
    create_access_token,
    SECRET_KEY,
    ALGORITHM
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Driver Drowsiness Detection API",
    description="Real-time drowsiness detection system with history tracking",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
init_db()

# Load drowsiness detection model
MODEL_PATH = os.path.join("model", "driver_drowsiness_final_model.keras")
detector = DrowsinessDetector(MODEL_PATH)

# Store active websocket connections
active_connections = {}

# Drowsiness tracking
drowsy_counters = {}  # Track consecutive drowsy frames per session
DROWSY_THRESHOLD = 15  # Trigger alert after 15 consecutive drowsy frames (~0.5 sec at 30fps)

# ==================== WebSocket Connection Manager ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"Client connected: {session_id}")
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Client disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)

manager = ConnectionManager()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_user_from_token(token: str, db: Session):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.username == username).first()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    user = get_user_from_token(token, db)
    if user is None:
        raise credentials_exception
    return user

# ==================== Startup Event ====================
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Driver Drowsiness Detection API")
    logger.info(f"Model loaded: {MODEL_PATH}")

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "Driver Drowsiness Detection API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/{session_id}",
            "history": "/api/history",
            "session": "/api/session",
            "stats": "/api/stats/{session_id}"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector.model is not None}

# ==================== Authentication ====================

@app.post("/api/auth/signup", response_model=UserResponse)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(
        or_(User.username == user_data.username, User.email == user_data.email)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
    
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=get_password_hash(user_data.password)
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@app.post("/api/auth/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# ==================== Session Management ====================

@app.post("/api/session/start", response_model=SessionResponse)
async def start_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Start a new detection session"""
    session_id = str(uuid.uuid4())
    
    new_session = SessionInfo(
        session_id=session_id,
        start_time=datetime.utcnow(),
        status="active",
        user_id=current_user.id
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    drowsy_counters[session_id] = 0
    
    logger.info(f"New session started: {session_id}")
    return new_session

@app.post("/api/session/end/{session_id}")
async def end_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """End an active session"""
    session = db.query(SessionInfo).filter(
        SessionInfo.session_id == session_id,
        SessionInfo.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.end_time = datetime.utcnow()
    session.status = "completed"
    db.commit()
    
    if session_id in drowsy_counters:
        del drowsy_counters[session_id]
    
    logger.info(f"Session ended: {session_id}")
    return {"message": "Session ended successfully", "session_id": session_id}

@app.get("/api/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get session details"""
    session = db.query(SessionInfo).filter(
        SessionInfo.session_id == session_id,
        SessionInfo.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@app.get("/api/session", response_model=List[SessionResponse])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List detection sessions for the current user"""
    sessions = db.query(SessionInfo).filter(
        SessionInfo.user_id == current_user.id
    ).order_by(SessionInfo.start_time.desc()).all()
    
    return sessions

# ==================== Statistics ====================

@app.get("/api/stats/{session_id}", response_model=SessionStats)
async def get_session_stats(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get statistics for a specific session"""
    session = db.query(SessionInfo).filter(
        SessionInfo.session_id == session_id,
        SessionInfo.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    total_detections = db.query(DetectionHistory).filter(
        DetectionHistory.session_id == session_id
    ).count()
    
    drowsy_count = db.query(DetectionHistory).filter(
        DetectionHistory.session_id == session_id,
        DetectionHistory.status == "drowsy"
    ).count()
    
    drowsy_percentage = (drowsy_count / total_detections * 100) if total_detections > 0 else 0
    
    return SessionStats(
        session_id=session.session_id,
        user_id=session.user_id,
        start_time=session.start_time,
        end_time=session.end_time,
        total_alerts=session.total_alerts,
        total_drowsy_time=session.total_drowsy_time,
        total_detections=total_detections,
        drowsy_percentage=round(drowsy_percentage, 2),
        status=session.status
    )

# ==================== History ====================

@app.get("/api/history", response_model=HistoryResponse)
async def get_history(
    session_id: str = None,
    limit: int = 100,
    skip: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detection history"""
    query = db.query(DetectionHistory).filter(
        DetectionHistory.user_id == current_user.id
    )
    
    if session_id:
        session = db.query(SessionInfo).filter(
            SessionInfo.session_id == session_id,
            SessionInfo.user_id == current_user.id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        query = query.filter(DetectionHistory.session_id == session_id)
    
    total = query.count()
    items = query.order_by(DetectionHistory.timestamp.desc()).offset(skip).limit(limit).all()
    
    return HistoryResponse(total=total, items=items)

@app.get("/api/history/{session_id}/alerts")
async def get_alerts(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all alert events for a session"""
    session = db.query(SessionInfo).filter(
        SessionInfo.session_id == session_id,
        SessionInfo.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    alerts = db.query(DetectionHistory).filter(
        DetectionHistory.session_id == session_id,
        DetectionHistory.user_id == current_user.id,
        DetectionHistory.alert_triggered == True
    ).order_by(DetectionHistory.timestamp.desc()).all()
    
    return {"session_id": session_id, "total_alerts": len(alerts), "alerts": alerts}

# ==================== WebSocket for Real-time Detection ====================

def decode_frame(frame_data: str) -> np.ndarray | None:
    """Decode base64 frame that may or may not include data URL prefix."""
    try:
        if "," in frame_data:
            base64_str = frame_data.split(",", 1)[1]
        else:
            base64_str = frame_data
        frame_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, db: Session = Depends(get_db)):
    token = websocket.query_params.get("token")
    user = get_user_from_token(token, db) if token else None
    
    if not user:
        await websocket.close(code=1008)
        return
    
    session = db.query(SessionInfo).filter(
        SessionInfo.session_id == session_id,
        SessionInfo.user_id == user.id
    ).first()
    
    if not session:
        await websocket.close(code=1008)
        return
    
    await manager.connect(websocket, session_id)
    
    drowsy_counters[session_id] = 0
    drowsy_start_time = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "frame":
                frame_data = message.get("frame")
                frame = decode_frame(frame_data)
                if frame is None:
                    continue
                
                # Predict (with face crop inside model_loader)
                result = detector.predict_from_frame(frame)
                
                status = result.get("status", "error")
                confidence = float(result.get("confidence", 0.0))
                face_coords = result.get("face_coords")  # (x, y, w, h) or None
                face_detected = face_coords is not None and status != "no_face"
                
                alert_triggered = False
                
                if status == "drowsy":
                    drowsy_counters[session_id] += 1
                    
                    if drowsy_start_time is None:
                        drowsy_start_time = datetime.utcnow()
                    
                    if drowsy_counters[session_id] >= DROWSY_THRESHOLD:
                        alert_triggered = True
                        
                        if drowsy_start_time:
                            duration = (datetime.utcnow() - drowsy_start_time).total_seconds()
                        else:
                            duration = 0
                        
                        session.total_alerts += 1
                        session.total_drowsy_time += duration
                        db.commit()
                        
                        drowsy_counters[session_id] = 0
                        drowsy_start_time = None
                else:
                    drowsy_counters[session_id] = 0
                    drowsy_start_time = None
                
                detection = DetectionHistory(
                    session_id=session_id,
                    user_id=user.id,
                    timestamp=datetime.utcnow(),
                    status=status,
                    confidence=confidence,
                    eye_status="closed" if status == "drowsy" else "open",
                    alert_triggered=alert_triggered
                )
                db.add(detection)
                db.commit()
                
                # ✅ FIX: Convert numpy int32 to Python int for JSON serialization
                face_coords_serializable = None
                if face_coords is not None:
                    face_coords_serializable = [int(coord) for coord in face_coords]
                
                response = {
                    "type": "detection",
                    "status": status,
                    "confidence": round(confidence, 4),
                    "alert": alert_triggered,
                    "drowsy_count": drowsy_counters[session_id],
                    "face_detected": face_detected,
                    "face_coords": face_coords_serializable,  # ✅ Now JSON serializable!
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await manager.send_message(session_id, response)
    
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        if session_id in drowsy_counters:
            del drowsy_counters[session_id]
        logger.info(f"WebSocket disconnected: {session_id}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(session_id)
        if session_id in drowsy_counters:
            del drowsy_counters[session_id]

# ==================== Image Upload Endpoint (Optional) ====================

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict drowsiness from uploaded image"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        result = detector.predict_from_frame(image)
        status = result.get("status", "error")
        confidence = float(result.get("confidence", 0.0))
        face_coords = result.get("face_coords")
        face_detected = face_coords is not None and status != "no_face"
        
        return {
            "status": status,
            "confidence": round(confidence, 4),
            "face_detected": face_detected,
            "face_coords": face_coords
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
