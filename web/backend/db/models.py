from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from db.database import Base

class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    target_role = Column(String)
    current_phase = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    emotion_records = relationship("EmotionRecord", back_populates="session", cascade="all, delete-orphan")
    skill_records = relationship("SkillRecord", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    intent = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="messages")

class EmotionRecord(Base):
    __tablename__ = "emotion_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    overall_state = Column(String(20))
    current_mood = Column(String(50))
    emotions = Column(Text)
    confidence = Column(Float)
    demand_type = Column(String(50))
    support_intensity = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="emotion_records")

class SkillRecord(Base):
    __tablename__ = "skill_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    skill_name = Column(String(100), nullable=False)
    level = Column(Integer)
    required_level = Column(Integer)
    category = Column(String(50))
    source = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    session = relationship("Session", back_populates="skill_records")
