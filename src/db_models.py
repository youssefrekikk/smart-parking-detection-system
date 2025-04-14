from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, JSON, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import uuid
from database import Base

class CameraStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"

class Camera(Base):
    __tablename__ = "cameras"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    stream_url = Column(String, nullable=True)
    location = Column(String, nullable=False)
    type = Column(String, nullable=False)  # webcam, ip_camera, video_file
    video_path = Column(String, nullable=True)
    status = Column(String, default="active")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    annotations = relationship("Annotation", back_populates="camera", cascade="all, delete-orphan")
    parking_spots = relationship("ParkingSpot", back_populates="camera", cascade="all, delete-orphan")
    frame_captures = relationship("FrameCapture", back_populates="camera", cascade="all, delete-orphan")
    occupancy_logs = relationship("OccupancyLog", back_populates="camera", cascade="all, delete-orphan")

class Annotation(Base):
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    mask_data = Column(JSON, nullable=False)  # Store the entire mask JSON
    reference_frame = Column(String, nullable=True)  # Path to reference frame image
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    camera = relationship("Camera", back_populates="annotations")
    parking_spots = relationship("ParkingSpot", back_populates="annotation", cascade="all, delete-orphan")

class ParkingGroup(Base):
    __tablename__ = "parking_groups"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    location = Column(String, nullable=False)
    coordinates = Column(JSON, nullable=True)  # Store location coordinates as JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    parking_spots = relationship("ParkingSpot", back_populates="group")
    occupancy_logs = relationship("OccupancyLog", back_populates="group", cascade="all, delete-orphan")

class ParkingSpot(Base):
    __tablename__ = "parking_spots"
    
    id = Column(Integer, primary_key=True)
    annotation_id = Column(Integer, ForeignKey("annotations.id", ondelete="CASCADE"), nullable=False)
    camera_id = Column(String, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(String, ForeignKey("parking_groups.id", ondelete="SET NULL"), nullable=True)
    spot_index = Column(Integer, nullable=False)  # Index within the annotation
    x_center = Column(Float, nullable=False)
    y_center = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    angle = Column(Float, default=0)
    corners = Column(JSON, nullable=True)  # Store corner points as JSON
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # Relationships
    annotation = relationship("Annotation", back_populates="parking_spots")
    camera = relationship("Camera", back_populates="parking_spots")
    group = relationship("ParkingGroup", back_populates="parking_spots")

class FrameCapture(Base):
    __tablename__ = "frame_captures"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    captured_at = Column(DateTime(timezone=True), server_default=func.now())
    frame_path = Column(String, nullable=True)  # Path to stored frame image
    
    # Relationships
    camera = relationship("Camera", back_populates="frame_captures")

class OccupancyLog(Base):
    __tablename__ = "occupancy_logs"
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(String, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(String, ForeignKey("parking_groups.id", ondelete="CASCADE"), nullable=False)
    captured_at = Column(DateTime(timezone=True), server_default=func.now())
    free_spots = Column(Integer, nullable=False)
    occupied_spots = Column(Integer, nullable=False)
    total_spots = Column(Integer, nullable=False)
    
    # Relationships
    camera = relationship("Camera", back_populates="occupancy_logs")
    group = relationship("ParkingGroup", back_populates="occupancy_logs")
