from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is not set. "
        "Please create a .env file with DATABASE_URL=postgresql://user:password@localhost:5432/drowsiness_db"
    )

# Validate PostgreSQL URL
if not DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg2://")):
    raise ValueError(
        f"Invalid DATABASE_URL. Expected PostgreSQL URL starting with 'postgresql://', "
        f"got: {DATABASE_URL[:20]}..."
    )

# Create engine with connection pooling for production
engine = create_engine(
    DATABASE_URL,
    pool_size=10,                    # Maximum number of permanent connections
    max_overflow=20,                 # Maximum number of temporary connections
    pool_timeout=30,                 # Timeout for getting connection from pool
    pool_recycle=3600,               # Recycle connections after 1 hour
    pool_pre_ping=True,              # Verify connections before using them
    echo=False,                      # Set to True for SQL query debugging
    future=True                      # Use SQLAlchemy 2.0 style
)

# Session local
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine,
    expire_on_commit=False           # Prevent lazy loading issues
)

# Base class for models
Base = declarative_base()

# ==================== Database Models ====================

class User(Base):
    """Application users"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    sessions = relationship("SessionInfo", back_populates="user", cascade="all, delete-orphan")
    detections = relationship("DetectionHistory", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"


class SessionInfo(Base):
    """Store session information"""
    __tablename__ = "session_info"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    total_alerts = Column(Integer, default=0, nullable=False)
    total_drowsy_time = Column(Float, default=0.0, nullable=False)  # in seconds
    status = Column(String(20), default="active", nullable=False)  # active, completed
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    detections = relationship("DetectionHistory", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<SessionInfo(id={self.id}, session_id='{self.session_id}', status='{self.status}')>"


class DetectionHistory(Base):
    """Store drowsiness detection history"""
    __tablename__ = "detection_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), ForeignKey("session_info.session_id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    status = Column(String(20), nullable=False)  # "alert", "drowsy", "no_face"
    confidence = Column(Float, nullable=False)  # Model confidence score
    eye_status = Column(String(20), nullable=True)  # "open" or "closed"
    duration_seconds = Column(Float, nullable=True)  # Duration of drowsiness
    alert_triggered = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="detections")
    session = relationship("SessionInfo", back_populates="detections")
    
    def __repr__(self):
        return f"<DetectionHistory(id={self.id}, status='{self.status}', alert={self.alert_triggered})>"


# ==================== Database Helper Functions ====================

def get_db():
    """
    Dependency to get database session
    Usage in FastAPI: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables
    Call this on application startup
    """
    try:
        logger.info("Initializing database...")
        logger.info(f"Database URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'hidden'}")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ Database tables created successfully!")
        logger.info(f"Tables created: {', '.join(Base.metadata.tables.keys())}")
        
        # Test connection
        with engine.connect() as connection:
            logger.info("✅ Database connection test successful!")
            
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        logger.error("Please check your DATABASE_URL and ensure PostgreSQL is running")
        raise


def drop_all_tables():
    """
    Drop all tables - USE WITH CAUTION!
    Only for development/testing
    """
    logger.warning("⚠️  Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("✅ All tables dropped")


def reset_database():
    """
    Reset database - drop and recreate all tables
    Only for development/testing
    """
    logger.warning("⚠️  Resetting database...")
    drop_all_tables()
    init_db()
    logger.info("✅ Database reset complete")


# ==================== Database Utility Functions ====================

def get_user_by_username(db, username: str):
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_email(db, email: str):
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_session_by_id(db, session_id: str):
    """Get session by session_id"""
    return db.query(SessionInfo).filter(SessionInfo.session_id == session_id).first()


def get_active_sessions(db, user_id: int):
    """Get all active sessions for a user"""
    return db.query(SessionInfo).filter(
        SessionInfo.user_id == user_id,
        SessionInfo.status == "active"
    ).all()


def cleanup_stale_sessions(db, hours: int = 24):
    """
    Cleanup sessions that are still 'active' but haven't been updated in X hours
    Call this periodically
    """
    from datetime import timedelta
    stale_time = datetime.utcnow() - timedelta(hours=hours)
    
    stale_sessions = db.query(SessionInfo).filter(
        SessionInfo.status == "active",
        SessionInfo.start_time < stale_time
    ).all()
    
    for session in stale_sessions:
        session.status = "completed"
        session.end_time = session.start_time + timedelta(hours=1)  # Estimate end time
    
    db.commit()
    logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")
    return len(stale_sessions)


# ==================== Connection Health Check ====================

def check_database_health():
    """
    Check if database connection is healthy
    Returns: (bool, str) - (is_healthy, message)
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True, "Database connection healthy"
    except Exception as e:
        return False, f"Database connection failed: {str(e)}"


# ==================== Startup Validation ====================

def validate_database_setup():
    """
    Validate database setup on startup
    Raises detailed errors if something is wrong
    """
    try:
        # Check if DATABASE_URL is set
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        # Check if it's PostgreSQL
        if not DATABASE_URL.startswith(("postgresql://", "postgresql+psycopg2://")):
            raise ValueError(f"Expected PostgreSQL URL, got: {DATABASE_URL[:20]}...")
        
        # Try to connect
        with engine.connect() as connection:
            result = connection.execute("SELECT version()")
            version = result.fetchone()[0]
            logger.info(f"✅ Connected to: {version}")
        
        # Check if tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            logger.warning("⚠️  No tables found. Run init_db() to create tables.")
        else:
            logger.info(f"✅ Found tables: {', '.join(tables)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database validation failed: {e}")
        raise


# ==================== Module Initialization ====================

# Log database configuration on import
logger.info("=" * 60)
logger.info("Database Configuration Loaded")
logger.info(f"Database Type: PostgreSQL")
logger.info(f"Host: {DATABASE_URL.split('@')[1].split('/')[0] if '@' in DATABASE_URL else 'unknown'}")
logger.info(f"Pool Size: 10 (max_overflow: 20)")
logger.info("=" * 60)