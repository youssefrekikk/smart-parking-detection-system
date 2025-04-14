from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Optional, Any, Union
import uuid
import json
import logging
from datetime import datetime

from db_models import Camera, Annotation, ParkingSpot, ParkingGroup, FrameCapture, OccupancyLog

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraRepository:
    """Repository for Camera operations"""
    
    @staticmethod
    def create_camera(db: Session, camera_data: Dict[str, Any]) -> Camera:
        """Create a new camera"""
        try:
            # Generate ID if not provided
            if 'id' not in camera_data or not camera_data['id']:
                camera_data['id'] = str(uuid.uuid4())
                
            camera = Camera(
                id=camera_data['id'],
                name=camera_data.get('name', f"Camera {camera_data['id']}"),
                stream_url=camera_data.get('url'),
                location=camera_data['location'],
                type=camera_data['type'],
                video_path=camera_data.get('video_path'),
                status=camera_data.get('status', 'active')
            )
            # Log the camera object before adding to session
            logger.info(f"Creating camera: {camera.id}, {camera.name}, {camera.type}")
            db.add(camera)
            try:
                db.flush()
                logger.info("Database flush successful")
            except Exception as flush_error:
                logger.error(f"Error during database flush: {str(flush_error)}")
                db.rollback()
                raise
            db.commit()
            db.refresh(camera)
            return camera
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating camera: {str(e)}")
            # Log more details about the error
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    @staticmethod
    def get_camera(db: Session, camera_id: str) -> Optional[Camera]:
        """Get a camera by ID"""
        return db.query(Camera).filter(Camera.id == camera_id).first()
    
    @staticmethod
    def get_all_cameras(db: Session) -> List[Camera]:
        """Get all cameras"""
        return db.query(Camera).all()
    
    @staticmethod
    def update_camera(db: Session, camera_id: str, camera_data: Dict[str, Any]) -> Optional[Camera]:
        """Update a camera"""
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                return None
                
            # Update fields
            for key, value in camera_data.items():
                if hasattr(camera, key) and key != 'id':
                    setattr(camera, key, value)
            
            camera.updated_at = datetime.now()
            db.commit()
            db.refresh(camera)
            return camera
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error updating camera: {str(e)}")
            raise
    
    @staticmethod
    def delete_camera(db: Session, camera_id: str) -> bool:
        """Delete a camera"""
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                return False
                
            db.delete(camera)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error deleting camera: {str(e)}")
            raise

class AnnotationRepository:
    """Repository for Annotation operations"""
    
    @staticmethod
    def create_annotation(db: Session, camera_id: str, mask_data: Dict[str, Any], reference_frame: str = None) -> Annotation:
        """Create a new annotation for a camera"""
        try:
            # Create annotation
            annotation = Annotation(
                camera_id=camera_id,
                mask_data=mask_data,
                reference_frame=reference_frame
            )
            db.add(annotation)
            db.flush()  # Get the ID without committing
            
            # Extract boxes from mask data
            boxes = mask_data.get("boxes", [])
            if not boxes and isinstance(mask_data, list):
                boxes = mask_data  # Handle case where mask_data is directly the boxes array
            
            # Create parking spots for this annotation
            for i, box in enumerate(boxes):
                try:
                    # Extract coordinates
                    if 'x1' in box and 'y1' in box and 'x2' in box and 'y2' in box:
                        # Calculate center, width, height
                        x_center = (box['x1'] + box['x2']) / 2
                        y_center = (box['y1'] + box['y2']) / 2
                        width = abs(box['x2'] - box['x1'])
                        height = abs(box['y2'] - box['y1'])
                    elif 'x_center' in box and 'y_center' in box and 'width' in box and 'height' in box:
                        x_center = box['x_center']
                        y_center = box['y_center']
                        width = box['width']
                        height = box['height']
                    else:
                        logger.warning(f"Box {i} has unsupported format: {box}")
                        continue
                    
                    # Get group ID from different possible keys
                    group_id = box.get('groupId') or box.get('group_id')
                    if group_id:
                        # Convert numeric IDs to strings if needed
                        group_id = str(group_id)
                    
                    # Create parking spot
                    spot = ParkingSpot(
                        annotation_id=annotation.id,
                        camera_id=camera_id,
                        group_id=group_id,
                        spot_index=i,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        angle=box.get('angle', 0),
                        corners=box.get('corners')
                    )
                    
                    db.add(spot)
                except Exception as e:
                    logger.error(f"Error creating parking spot {i}: {str(e)}")
                    # Continue with next spot
            
            # Commit all changes
            db.commit()
            db.refresh(annotation)
            return annotation
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating annotation: {str(e)}")
            raise

    @staticmethod
    def get_latest_annotation(db: Session, camera_id: str) -> Optional[Annotation]:
        """Get the latest annotation for a camera"""
        return db.query(Annotation)\
                .filter(Annotation.camera_id == camera_id)\
                .order_by(Annotation.created_at.desc())\
                .first()
    
    @staticmethod
    def get_annotation(db: Session, annotation_id: int) -> Optional[Annotation]:
        """Get an annotation by ID"""
        return db.query(Annotation).filter(Annotation.id == annotation_id).first()

class ParkingSpotRepository:
    """Repository for ParkingSpot operations"""
    
    @staticmethod
    def create_spots_from_mask(db: Session, annotation_id: int, camera_id: str, 
                            mask_data: Dict[str, Any]) -> List[ParkingSpot]:
        """Create parking spots from mask data"""
        try:
            spots = []
            # Handle both formats - direct boxes array or nested in 'boxes' key
            boxes = mask_data.get('boxes', [])
            if not boxes and isinstance(mask_data, list):
                boxes = mask_data  # Handle case where mask_data is directly the boxes array
            
            logger.info(f"Processing {len(boxes)} parking spots for camera {camera_id}, annotation {annotation_id}")
            
            for i, box in enumerate(boxes):
                try:
                    # Extract coordinates
                    if 'x1' in box and 'y1' in box and 'x2' in box and 'y2' in box:
                        # Calculate center, width, height
                        x_center = (box['x1'] + box['x2']) / 2
                        y_center = (box['y1'] + box['y2']) / 2
                        width = abs(box['x2'] - box['x1'])
                        height = abs(box['y2'] - box['y1'])
                    elif 'x_center' in box and 'y_center' in box and 'width' in box and 'height' in box:
                        x_center = box['x_center']
                        y_center = box['y_center']
                        width = box['width']
                        height = box['height']
                    else:
                        logger.warning(f"Box {i} has unsupported format: {box}")
                        continue
                    
                    # Get group ID from different possible keys
                    group_id = box.get('groupId') or box.get('group_id')
                    if group_id:
                        # Convert numeric IDs to strings if needed
                        group_id = str(group_id)
                    
                    # Create parking spot
                    spot = ParkingSpot(
                        annotation_id=annotation_id,
                        camera_id=camera_id,
                        group_id=group_id,
                        spot_index=i,
                        x_center=x_center,
                        y_center=y_center,
                        width=width,
                        height=height,
                        angle=box.get('angle', 0),
                        corners=box.get('corners')
                    )
                    
                    logger.info(f"Creating spot {i} for camera {camera_id}, group {group_id}")
                    db.add(spot)
                    spots.append(spot)
                except Exception as box_error:
                    logger.error(f"Error processing box {i}: {str(box_error)}")
                    continue
            
            db.commit()
            for spot in spots:
                db.refresh(spot)
                
            return spots
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating parking spots: {str(e)}")
            raise

    @staticmethod
    def get_spots_by_camera(db: Session, camera_id: str) -> List[ParkingSpot]:
        """Get all parking spots for a camera"""
        return db.query(ParkingSpot).filter(ParkingSpot.camera_id == camera_id).all()
    
    @staticmethod
    def get_spots_by_group(db: Session, group_id: str) -> List[ParkingSpot]:
        """Get all parking spots in a group"""
        return db.query(ParkingSpot).filter(ParkingSpot.group_id == group_id).all()

class ParkingGroupRepository:
    """Repository for ParkingGroup operations"""
    
    @staticmethod
    def create_groups_from_mask(db: Session, camera_id: str, mask_data: Dict[str, Any]) -> List[ParkingGroup]:
        """Create parking groups from mask data"""
        try:
            groups = []
            # Handle both formats - direct groups array or nested in 'groups' key
            group_data = mask_data.get('groups', [])
            
            # Add debug logging
            logger.info(f"Processing {len(group_data)} groups for camera {camera_id}")
            logger.info(f"Group data sample: {group_data[:1] if group_data else 'None'}")
            
            for group_info in group_data:
                # Handle different group ID formats
                group_id = None
                if isinstance(group_info, dict):
                    group_id = group_info.get('group_id') or group_info.get('id')
                    
                if not group_id:
                    group_id = str(uuid.uuid4())
                    logger.info(f"Generated new group ID: {group_id}")
                else:
                    # Convert numeric IDs to strings if needed
                    group_id = str(group_id)
                    logger.info(f"Using existing group ID: {group_id}")
                
                # Get location from different possible keys
                location = None
                if isinstance(group_info, dict):
                    location = group_info.get('location') or group_info.get('name', 'Unknown')
                
                # Check if group already exists
                existing_group = db.query(ParkingGroup).filter(ParkingGroup.id == group_id).first()
                if existing_group:
                    logger.info(f"Group {group_id} already exists, skipping creation")
                    groups.append(existing_group)
                    continue
                
                # Create new group
                group = ParkingGroup(
                    id=group_id,
                    name=group_info.get('name') if isinstance(group_info, dict) else None,
                    location=location or 'Unknown',
                    coordinates=group_info.get('coordinates') if isinstance(group_info, dict) else None
                )
                
                logger.info(f"Creating new group: {group_id}, location: {location}")
                db.add(group)
                groups.append(group)
            
            db.commit()
            for group in groups:
                db.refresh(group)
                
            return groups
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error creating parking groups: {str(e)}")
            raise

    @staticmethod
    def get_all_groups(db: Session) -> List[ParkingGroup]:
        """Get all parking groups"""
        return db.query(ParkingGroup).all()
    
    @staticmethod
    def get_group(db: Session, group_id: str) -> Optional[ParkingGroup]:
        """Get a parking group by ID"""
        return db.query(ParkingGroup).filter(ParkingGroup.id == group_id).first()

class OccupancyRepository:
    """Repository for OccupancyLog operations"""
    
    @staticmethod
    def log_occupancy(db: Session, camera_id: str, group_id: str, free_spots: int, 
                    occupied_spots: int, total_spots: int) -> OccupancyLog:
        """Log occupancy data for a parking group"""
        try:
            # Convert group_id to string if it's not already
            group_id = str(group_id)
            
            # Check if group exists
            group = db.query(ParkingGroup).filter(ParkingGroup.id == group_id).first()
            if not group:
                # Create a placeholder group if it doesn't exist
                logger.warning(f"Group {group_id} not found, creating placeholder")
                group = ParkingGroup(
                    id=group_id,
                    name=f"Group {group_id}",
                    location="Auto-created"
                )
                db.add(group)
                db.flush()
            
            # Create occupancy log
            occupancy = OccupancyLog(
                camera_id=camera_id,
                group_id=group_id,
                free_spots=free_spots,
                occupied_spots=occupied_spots,
                total_spots=total_spots
            )
            
            db.add(occupancy)
            db.commit()
            db.refresh(occupancy)
            return occupancy
        except Exception as e:
            db.rollback()
            logger.error(f"Error logging occupancy: {str(e)}")
            raise

    @staticmethod
    def get_latest_occupancy(db: Session, group_id: str) -> Optional[OccupancyLog]:
        """Get the latest occupancy log for a group"""
        return db.query(OccupancyLog)\
                .filter(OccupancyLog.group_id == group_id)\
                .order_by(OccupancyLog.captured_at.desc())\
                .first()
    
    @staticmethod
    def get_occupancy_history(db: Session, group_id: str, limit: int = 24) -> List[OccupancyLog]:
        """Get occupancy history for a group"""
        return db.query(OccupancyLog)\
                .filter(OccupancyLog.group_id == group_id)\
                .order_by(OccupancyLog.captured_at.desc())\
                .limit(limit)\
                .all()

class FrameCaptureRepository:
    """Repository for FrameCapture operations"""
    
    @staticmethod
    def save_frame(db: Session, camera_id: str, frame_path: str) -> FrameCapture:
        """Save a captured frame"""
        try:
            frame = FrameCapture(
                camera_id=camera_id,
                frame_path=frame_path
            )
            
            db.add(frame)
            db.commit()
            db.refresh(frame)
            return frame
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Error saving frame: {str(e)}")
            raise
