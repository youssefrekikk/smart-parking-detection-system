from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Optional, Any
import uvicorn
import json
import asyncio
import cv2
import numpy as np
import os
os.environ["TORCH_ALLOW_WEIGHTS_ONLY_SKIP"] = "1"
import shutil
from sqlalchemy.orm import Session
import base64
import logging
from datetime import datetime
from parking_spot_annotation_model import ParkingSpotAnnotator
import tempfile
from fastapi.responses import JSONResponse

# Import local modules
from video_simulator import VideoSimulator
from database import get_db
from repositories import (
    CameraRepository, AnnotationRepository, ParkingSpotRepository,
    ParkingGroupRepository, OccupancyRepository, FrameCaptureRepository
)
from db_models import ParkingGroup


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import parking detection modules with fallback
try:
    from parking_spot_detection import load_parking_spots_from_json, detect_parking_spots, fallback_detection
    parking_detection_available = True
except ImportError:
    logger.warning("parking_spot_detection module not found. Parking detection features will be disabled.")
    parking_detection_available = False

app = FastAPI(title="Video Processing Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("masks", exist_ok=True)

# Store active camera connections and their metadata (in-memory cache)
active_cameras: Dict[str, Dict] = {}

# Store video simulators for each camera
video_simulators: Dict[str, VideoSimulator] = {}

@app.on_event("startup")
async def startup_event():
    """Load active cameras from database on startup"""
    db = next(get_db())
    try:
        # Load all active cameras from database
        cameras = CameraRepository.get_all_cameras(db)
        for camera in cameras:
            # Add to active_cameras cache
            active_cameras[camera.id] = {
                "url": camera.stream_url,
                "location": camera.location,
                "type": camera.type,
                "video_path": camera.video_path,
                "status": camera.status
            }
            
            # Initialize video simulator for video files
            if camera.type == "video_file" and camera.video_path and os.path.exists(camera.video_path):
                try:
                    simulator = VideoSimulator(camera.video_path)
                    simulator.start()
                    video_simulators[camera.id] = simulator
                    logger.info(f"Video simulator started for camera {camera.id}")
                except Exception as e:
                    logger.error(f"Error starting video simulator for camera {camera.id}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading cameras from database: {str(e)}")
    finally:
        db.close()

@app.get("/")
async def root():
    return FileResponse("src/static/admin.html")

@app.get("/admin")
async def admin():
    return FileResponse("src/static/admin.html")

@app.post("/register-camera")
async def register_camera(
    id: str = Form(...),
    url: str = Form(None),  # Make URL optional
    type: str = Form(...),
    location: str = Form(...),
    video_file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """Register a new camera with its metadata"""
    try:
        camera_id = id
        if not camera_id:
            raise HTTPException(status_code=400, detail="Camera ID is required")
        
        logger.info(f"Registering camera {camera_id}...")
        
        # Handle video file upload if provided
        video_path = None
        if video_file:
            try:
                # Save uploaded video file
                video_path = f"uploads/{camera_id}_{video_file.filename}"
                with open(video_path, "wb") as buffer:
                    shutil.copyfileobj(video_file.file, buffer)
                logger.info(f"Video file saved to: {video_path}")
                
                # Initialize video simulator for this video
                simulator = VideoSimulator(video_path)
                try:
                    simulator.start()
                    video_simulators[camera_id] = simulator
                    logger.info(f"Video simulator started for camera {camera_id}")
                except Exception as e:
                    logger.error(f"Error starting video simulator: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to start video simulator: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error handling video file: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to process video file: {str(e)}")
        
        # Create camera data dictionary
        camera_data = {
            "id": camera_id,
            "name": f"Camera {camera_id}",
            "url": url,
            "location": location,
            "type": type,
            "video_path": video_path,
            "status": "active"
        }
        
        # Store camera in database
        try:
            camera = CameraRepository.create_camera(db, camera_data)
            logger.info(f"Camera {camera_id} saved to database")
        except Exception as e:
            logger.error(f"Error saving camera to database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save camera to database: {str(e)}")
        
        # Store camera metadata in memory cache
        active_cameras[camera_id] = {
            "url": url,
            "location": location,
            "type": type,
            "video_path": video_path
        }
        
        logger.info(f"Camera {camera_id} registered successfully")
        return {"message": f"Camera {camera_id} registered successfully"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in register_camera: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.websocket("/ws/{camera_id}")
async def websocket_endpoint(websocket: WebSocket, camera_id: str):
    """WebSocket endpoint for receiving video frames"""
    if camera_id not in active_cameras:
        await websocket.close(code=1008, reason="Camera not registered")
        return
        
    camera_info = active_cameras[camera_id]
    await websocket.accept()
    logger.info(f"WebSocket connection established for camera {camera_id}")
    
    cap = None
    try:
        # Check if we have a video simulator for this camera
        if camera_id in video_simulators:
            logger.info(f"Using video simulator for camera {camera_id}")
            simulator = video_simulators[camera_id]
            
            # Send a test frame to confirm connection works
            test_frame = simulator.get_frame()
            if test_frame is not None:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_bytes = buffer.tobytes()
                await websocket.send_bytes(frame_bytes)
                logger.info(f"Sent test frame to client for camera {camera_id}")
            
            while True:
                # Get frame from simulator
                frame = simulator.get_frame()
                if frame is None:
                    logger.warning("No frame available from simulator")
                    await asyncio.sleep(0.1)
                    continue
                
                # Resize frame if too large
                max_width = 1280
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_bytes = buffer.tobytes()
                
                try:
                    await websocket.send_bytes(frame_bytes)
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")
                    break
                
                # Control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
        else:
            # Initialize video capture based on camera type
            if camera_info["type"] == "webcam":
                logger.info(f"Opening webcam for camera {camera_id}")
                cap = cv2.VideoCapture(0)  # Use default webcam
                
            elif camera_info["type"] == "ip_camera":
                logger.info(f"Opening IP camera {camera_id} at {camera_info['url']}")
                cap = cv2.VideoCapture(camera_info["url"])
                
            elif camera_info["type"] == "video_file":
                logger.info(f"Opening video file for camera {camera_id}: {camera_info['video_path']}")
                cap = cv2.VideoCapture(camera_info["video_path"])
                
            if not cap or not cap.isOpened():
                error_msg = f"Failed to open video source for camera {camera_id}"
                logger.error(error_msg)
                await websocket.close(code=1011, reason=error_msg)
                return

            # Set some basic properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
            cap.set(cv2.CAP_PROP_FPS, 30)  # Try to set 30 FPS

            # Send a test frame to confirm connection works
            ret, test_frame = cap.read()
            if ret:
                # Resize frame if too large
                max_width = 1280
                if test_frame.shape[1] > max_width:
                    scale = max_width / test_frame.shape[1]
                    test_frame = cv2.resize(test_frame, None, fx=scale, fy=scale)
                
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', test_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_bytes = buffer.tobytes()
                await websocket.send_bytes(frame_bytes)
                logger.info(f"Sent test frame to client for camera {camera_id}")
                
                # Reset video position if it's a file
                if camera_info["type"] == "video_file":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    if camera_info["type"] == "video_file":
                        # Reset to beginning of video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.error(f"Failed to read frame from camera {camera_id}")
                    break

                # Resize frame if too large
                max_width = 1280
                if frame.shape[1] > max_width:
                    scale = max_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)

                # Convert frame to JPEG with better quality
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_bytes = buffer.tobytes()
                
                try:
                    await websocket.send_bytes(frame_bytes)
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")
                    break
                
                # Control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
            
    except Exception as e:
        logger.error(f"Error in websocket connection for camera {camera_id}: {str(e)}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        # Clean up resources
        if cap is not None:
            cap.release()
            logger.info(f"Released video capture for camera {camera_id}")
        
        # If we were using a simulator, don't stop it yet as we might need it again
        # The simulator will be cleaned up when the application shuts down
        logger.info(f"WebSocket connection closed for camera {camera_id}")

@app.post("/save-mask/{camera_id}")
async def save_mask(camera_id: str, data: Dict[str, Any], db: Session = Depends(get_db)):
    """Save mask boxes drawn by admin in enhanced format with group information"""
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        # Extract boxes and groups from the received data
        boxes = data.get("boxes", [])
        groups = data.get("groups", [])
        
        if not boxes:
            raise HTTPException(status_code=400, detail="No boxes provided")
        
        # First, create or update parking groups
        created_groups = []
        if groups:
            logger.info(f"Processing {len(groups)} groups for camera {camera_id}")
            for group in groups:
                group_id = str(group.get("group_id"))
                location = group.get("location", "Unknown")
                
                # Check if group exists
                existing_group = db.query(ParkingGroup).filter(ParkingGroup.id == group_id).first()
                
                if existing_group:
                    logger.info(f"Updating existing group {group_id}")
                    existing_group.location = location
                    db.add(existing_group)
                    created_groups.append(existing_group)
                else:
                    logger.info(f"Creating new group {group_id} with location {location}")
                    new_group = ParkingGroup(
                        id=group_id,
                        name=f"Group {location}",
                        location=location
                    )
                    db.add(new_group)
                    created_groups.append(new_group)
            
            # Commit group changes
            try:
                db.commit()
                logger.info(f"Successfully saved {len(created_groups)} groups to database")
            except Exception as e:
                db.rollback()
                logger.error(f"Error saving groups to database: {str(e)}")
                # Continue with annotation saving even if group saving fails
        
        # Save to database using repository
        try:
            # Get reference frame path if available
            reference_frame = None
            frame_captures = FrameCaptureRepository.save_frame(
                db, 
                camera_id, 
                f"masks/{camera_id}_reference_frame.jpg"
            )
            if frame_captures:
                reference_frame = frame_captures.frame_path
            
            # Create annotation in database
            annotation = AnnotationRepository.create_annotation(
                db, 
                camera_id, 
                data,  # Save the entire enhanced data structure
                reference_frame
            )
            
            logger.info(f"Annotation saved to database for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error saving annotation to database: {str(e)}")
            # Continue with file-based storage as fallback
        
        # Also save to file for backward compatibility
        mask_path = f"masks/{camera_id}_mask.json"
        with open(mask_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return {
            "message": "Mask saved successfully", 
            "boxes": len(boxes), 
            "groups": len(groups)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save mask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save mask: {str(e)}")

@app.get("/capture-frame/{camera_id}")
async def capture_frame(camera_id: str, db: Session = Depends(get_db)):
    """Capture a single frame from the specified camera"""
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
        
    camera_info = active_cameras[camera_id]
    
    try:
        # Check if we have a video simulator for this camera
        if camera_id in video_simulators:
            
            logger.info(f"Capturing frame from video simulator for camera {camera_id}")
            simulator = video_simulators[camera_id]
            frame = simulator.get_frame()
            
            if frame is None:
                raise Exception("Failed to capture frame from simulator")
        else:
            cap = None
            try:
                # Initialize video capture based on camera type
                if camera_info["type"] == "webcam":
                    cap = cv2.VideoCapture(0)
                elif camera_info["type"] == "ip_camera":
                    cap = cv2.VideoCapture(camera_info["url"])
                elif camera_info["type"] == "video_file":
                    cap = cv2.VideoCapture(camera_info["video_path"])
                    
                if not cap or not cap.isOpened():
                    raise Exception("Failed to open video source")
                    
                # Read a single frame
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to capture frame")
            finally:
                if cap is not None:
                    cap.release()
            
        # Resize frame if too large
        max_width = 1280
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Save frame to disk and database
        frame_path = f"uploads/{camera_id}_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Save frame record to database
        try:
            FrameCaptureRepository.save_frame(db, camera_id, frame_path)
            logger.info(f"Frame saved to database for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error saving frame to database: {str(e)}")
            # Continue with response even if database save fails
            
        # Convert frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_bytes = buffer.tobytes()
        
        return Response(content=frame_bytes, media_type="image/jpeg")
        
    except Exception as e:
        logger.error(f"Error capturing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Stop all video simulators when app is shutting down"""
    for camera_id, simulator in video_simulators.items():
        logger.info(f"Stopping video simulator for camera {camera_id}")
        simulator.stop()
        
# Add this route to serve the parking test page
@app.get("/parking-test")
async def parking_test_page():
    return FileResponse("src/static/parking-test.html")

# Add this endpoint for testing parking detection
@app.get("/test-parking-detection/{camera_id}")
async def test_parking_detection(camera_id: str, method: str = "yolo", db: Session = Depends(get_db)):
    """Test endpoint for parking spot detection"""
    if camera_id not in active_cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    try:
        # First try to get annotation from database
        annotation = None
        try:
            annotation = AnnotationRepository.get_latest_annotation(db, camera_id)
            if annotation:
                logger.info(f"Found annotation in database for camera {camera_id}")
                mask_data = annotation.mask_data
            else:
                logger.info(f"No annotation found in database for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error retrieving annotation from database: {str(e)}")
            # Continue with file-based approach as fallback
        
        # Fallback to file-based mask if database retrieval failed
        mask_path = f"masks/{camera_id}_mask.json"
        if not annotation and not os.path.exists(mask_path):
            raise HTTPException(status_code=404, detail="No mask found for this camera")
        
        # Get a frame from the camera
        frame = None
        
        # Check if we have a video simulator for this camera
        if camera_id in video_simulators:
            simulator = video_simulators[camera_id]
            frame = simulator.get_frame()
        else:
            # Capture a frame from the camera
            cap = None
            try:
                camera_info = active_cameras[camera_id]
                if camera_info["type"] == "webcam":
                    cap = cv2.VideoCapture(0)
                elif camera_info["type"] == "ip_camera":
                    cap = cv2.VideoCapture(camera_info["url"])
                elif camera_info["type"] == "video_file":
                    cap = cv2.VideoCapture(camera_info["video_path"])
                
                if not cap or not cap.isOpened():
                    raise Exception("Failed to open video source")
                
                ret, frame = cap.read()
                if not ret:
                    raise Exception("Failed to capture frame")
            finally:
                if cap is not None:
                    cap.release()
        
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to get frame from camera")
        
        # Load parking spots
        try:
            # If we have annotation from database, use it
            if annotation:
                # Convert mask_data to format expected by load_parking_spots_from_json
                # This is a temporary solution until we update the detection code
                with open("temp_mask.json", "w") as f:
                    json.dump(annotation.mask_data, f)
                parking_spots = load_parking_spots_from_json("temp_mask.json", frame.shape[1], frame.shape[0])
                os.remove("temp_mask.json")  # Clean up
            else:
                # Load from file as before
                parking_spots = load_parking_spots_from_json(mask_path, frame.shape[1], frame.shape[0])
            
            logger.info(f"Successfully loaded {len(parking_spots)} parking spots")
        except Exception as e:
            logger.error(f"Error loading parking spots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error loading parking spots: {str(e)}")
        
        # Run parking detection based on method
        try:
            if method == "fallback":
                results = fallback_detection(frame, parking_spots)
            else:
                try:
                    # Try YOLO detection first
                    results = detect_parking_spots(frame, parking_spots)
                except Exception as e:
                    logger.error(f"YOLO detection failed: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Fall back to simple detection if YOLO fails
                    results = fallback_detection(frame, parking_spots)
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
        
        # Save occupancy results to database
        try:
            # Process each group's occupancy
            for group_id, stats in results["group_stats"].items():
                OccupancyRepository.log_occupancy(
                    db,
                    camera_id,
                    group_id,
                    stats["available"],  # free spots
                    stats["total"] - stats["available"],  # occupied spots
                    stats["total"]  # total spots
                )
            logger.info(f"Occupancy data saved to database for camera {camera_id}")
        except Exception as e:
            logger.error(f"Error saving occupancy data to database: {str(e)}")
            # Continue with response even if database save fails
        
        # Convert visualization frame to base64 for response
        _, buffer = cv2.imencode('.jpg', results['visualization_frame'])
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Return results with visualization
        return {
            "total_spots": results["total_spots"],
            "available_spots": results["available_spots"],
            "percentage_available": results["percentage_available"],
            "spot_status": results["spot_status"],
            "group_stats": results["group_stats"],
            "visualization": img_str
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in test_parking_detection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/debug-mask/{camera_id}")
async def debug_mask(camera_id: str, db: Session = Depends(get_db)):
    """Debug endpoint to check the format of the saved mask file"""
    # First try to get from database
    try:
        annotation = AnnotationRepository.get_latest_annotation(db, camera_id)
        if annotation:
            return {
                "source": "database",
                "mask_data": annotation.mask_data,
                "annotation_id": annotation.id,
                "created_at": annotation.created_at
            }
    except Exception as e:
        logger.error(f"Error retrieving annotation from database: {str(e)}")
        # Fall back to file-based approach
    
    # Check file-based mask
    mask_path = f"masks/{camera_id}_mask.json"
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="No mask found for this camera")
    
    try:
        with open(mask_path, 'r') as f:
            mask_data = json.load(f)
        
        return {
            "source": "file",
            "mask_data": mask_data,
            "file_path": mask_path,
            "file_size": os.path.getsize(mask_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading mask file: {str(e)}")

@app.get("/list-cameras")
async def list_cameras(db: Session = Depends(get_db)):
    """List all registered cameras and available masks"""
    # Get cameras from database
    db_cameras = []
    try:
        db_cameras = CameraRepository.get_all_cameras(db)
        logger.info(f"Found {len(db_cameras)} cameras in database")
    except Exception as e:
        logger.error(f"Error retrieving cameras from database: {str(e)}")
    
    # Get all registered cameras from memory cache
    memory_cameras = list(active_cameras.keys())
    
    # Check for mask files
    mask_files = os.listdir("masks")
    available_masks = []
    
    for mask_file in mask_files:
        if mask_file.endswith("_mask.json"):
            camera_id = mask_file.split("_mask.json")[0]
            available_masks.append(camera_id)
    
    # Combine all sources
    all_camera_ids = set()
    
    # Add database cameras
    for camera in db_cameras:
        all_camera_ids.add(camera.id)
        if camera.id not in active_cameras:
            # Add to active_cameras cache if not already there
            active_cameras[camera.id] = {
                "url": camera.stream_url,
                "location": camera.location,
                "type": camera.type,
                "video_path": camera.video_path,
                "status": camera.status,
                "source": "database"
            }
    
    # Add memory-cached cameras
    for camera_id in memory_cameras:
        all_camera_ids.add(camera_id)
    
    # Add cameras with masks
    for camera_id in available_masks:
        all_camera_ids.add(camera_id)
        if camera_id not in active_cameras:
            # Add a placeholder entry
            active_cameras[camera_id] = {
                "type": "unknown",
                "location": "Unknown",
                "loaded_from_mask": True,
                "source": "mask_file"
            }
    
    return {
        "active_cameras": list(all_camera_ids),
        "cameras_with_masks": available_masks,
        "database_cameras": [camera.id for camera in db_cameras]
    }

@app.get("/camera/{camera_id}")
async def get_camera(camera_id: str, db: Session = Depends(get_db)):
    """Get camera details"""
    # Try to get from database first
    try:
        camera = CameraRepository.get_camera(db, camera_id)
        if camera:
            return {
                "id": camera.id,
                "name": camera.name,
                "url": camera.stream_url,
                "location": camera.location,
                "type": camera.type,
                "video_path": camera.video_path,
                "status": camera.status,
                "source": "database"
            }
    except Exception as e:
        logger.error(f"Error retrieving camera from database: {str(e)}")
    
    # Fall back to memory cache
    if camera_id in active_cameras:
        return {
            "id": camera_id,
            **active_cameras[camera_id],
            "source": active_cameras[camera_id].get("source", "memory")
        }
    
    raise HTTPException(status_code=404, detail="Camera not found")

@app.get("/occupancy/{group_id}")
async def get_occupancy(group_id: str, db: Session = Depends(get_db)):
    """Get latest occupancy data for a parking group"""
    try:
        occupancy = OccupancyRepository.get_latest_occupancy(db, group_id)
        if not occupancy:
            raise HTTPException(status_code=404, detail="No occupancy data found for this group")
        
        return {
            "group_id": occupancy.group_id,
            "camera_id": occupancy.camera_id,
            "free_spots": occupancy.free_spots,
            "occupied_spots": occupancy.occupied_spots,
            "total_spots": occupancy.total_spots,
            "percentage_available": round(occupancy.free_spots / max(occupancy.total_spots, 1) * 100, 1),
            "captured_at": occupancy.captured_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving occupancy data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving occupancy data: {str(e)}")

@app.get("/occupancy-history/{group_id}")
async def get_occupancy_history(group_id: str, limit: int = 24, db: Session = Depends(get_db)):
    """Get occupancy history for a parking group"""
    try:
        history = OccupancyRepository.get_occupancy_history(db, group_id, limit)
        if not history:
            raise HTTPException(status_code=404, detail="No occupancy history found for this group")
        
        return [
            {
                "group_id": entry.group_id,
                "camera_id": entry.camera_id,
                "free_spots": entry.free_spots,
                "occupied_spots": entry.occupied_spots,
                                "total_spots": entry.total_spots,
                "percentage_available": round(entry.free_spots / max(entry.total_spots, 1) * 100, 1),
                "captured_at": entry.captured_at
            }
            for entry in history
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving occupancy history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving occupancy history: {str(e)}")

@app.get("/groups")
async def get_all_groups(db: Session = Depends(get_db)):
    """Get all parking groups"""
    try:
        groups = ParkingGroupRepository.get_all_groups(db)
        return [
            {
                "id": group.id,
                "name": group.name,
                "location": group.location,
                "coordinates": group.coordinates
            }
            for group in groups
        ]
    except Exception as e:
        logger.error(f"Error retrieving parking groups: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving parking groups: {str(e)}")

@app.get("/group/{group_id}")
async def get_group(group_id: str, db: Session = Depends(get_db)):
    """Get a parking group by ID"""
    try:
        group = ParkingGroupRepository.get_group(db, group_id)
        if not group:
            raise HTTPException(status_code=404, detail="Group not found")
        
        # Get spots for this group
        spots = ParkingSpotRepository.get_spots_by_group(db, group_id)
        
        # Get latest occupancy
        occupancy = OccupancyRepository.get_latest_occupancy(db, group_id)
        
        return {
            "id": group.id,
            "name": group.name,
            "location": group.location,
            "coordinates": group.coordinates,
            "spots_count": len(spots),
            "latest_occupancy": {
                "free_spots": occupancy.free_spots if occupancy else None,
                "occupied_spots": occupancy.occupied_spots if occupancy else None,
                "total_spots": occupancy.total_spots if occupancy else None,
                "percentage_available": round(occupancy.free_spots / max(occupancy.total_spots, 1) * 100, 1) if occupancy else None,
                "captured_at": occupancy.captured_at if occupancy else None
            } if occupancy else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving parking group: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving parking group: {str(e)}")

@app.get("/spots/{camera_id}")
async def get_spots(camera_id: str, db: Session = Depends(get_db)):
    """Get all parking spots for a camera"""
    try:
        spots = ParkingSpotRepository.get_spots_by_camera(db, camera_id)
        return [
            {
                "id": spot.id,
                "camera_id": spot.camera_id,
                "group_id": spot.group_id,
                "spot_index": spot.spot_index,
                "x_center": spot.x_center,
                "y_center": spot.y_center,
                "width": spot.width,
                "height": spot.height,
                "angle": spot.angle,
                "corners": spot.corners
            }
            for spot in spots
        ]
    except Exception as e:
        logger.error(f"Error retrieving parking spots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving parking spots: {str(e)}")

@app.delete("/camera/{camera_id}")
async def delete_camera(camera_id: str, db: Session = Depends(get_db)):
    """Delete a camera and all associated data"""
    try:
        # Delete from database
        result = CameraRepository.delete_camera(db, camera_id)
        
        # Remove from memory cache
        if camera_id in active_cameras:
            del active_cameras[camera_id]
        
        # Stop and remove video simulator if exists
        if camera_id in video_simulators:
            video_simulators[camera_id].stop()
            del video_simulators[camera_id]
        
        # Delete mask file if exists
        mask_path = f"masks/{camera_id}_mask.json"
        if os.path.exists(mask_path):
            os.remove(mask_path)
        
        return {"message": f"Camera {camera_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting camera: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting camera: {str(e)}")

@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard page"""
    return FileResponse("src/static/dashboard.html")

@app.get("/dashboard-data")
async def dashboard_data(db: Session = Depends(get_db)):
    """Get data for the dashboard"""
    try:
        # Get all cameras
        cameras = CameraRepository.get_all_cameras(db)
        
        # Get all groups
        groups = ParkingGroupRepository.get_all_groups(db)
        
        # Get latest occupancy for each group
        group_occupancy = {}
        for group in groups:
            occupancy = OccupancyRepository.get_latest_occupancy(db, group.id)
            if occupancy:
                group_occupancy[group.id] = {
                    "free_spots": occupancy.free_spots,
                    "occupied_spots": occupancy.occupied_spots,
                    "total_spots": occupancy.total_spots,
                    "percentage_available": round(occupancy.free_spots / max(occupancy.total_spots, 1) * 100, 1),
                    "captured_at": occupancy.captured_at
                }
        
        return {
            "cameras": [
                {
                    "id": camera.id,
                    "name": camera.name,
                    "location": camera.location,
                    "type": camera.type,
                    "status": camera.status
                }
                for camera in cameras
            ],
            "groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "location": group.location,
                    "occupancy": group_occupancy.get(group.id)
                }
                for group in groups
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving dashboard data: {str(e)}")

# Data migration endpoint
@app.post("/migrate-data")
async def migrate_data(db: Session = Depends(get_db)):
    """Migrate existing JSON data to the database"""
    try:
        # Get all mask files
        mask_files = [f for f in os.listdir("masks") if f.endswith("_mask.json")]
        
        migrated_count = 0
        for mask_file in mask_files:
            camera_id = mask_file.split("_mask.json")[0]
            mask_path = f"masks/{mask_file}"
            
            # Check if camera exists in database
            camera = CameraRepository.get_camera(db, camera_id)
            if not camera:
                # Create camera if not exists
                camera_data = {
                    "id": camera_id,
                    "name": f"Camera {camera_id}",
                    "location": "Unknown",
                    "type": "unknown"
                }
                
                # Check if we have info in active_cameras
                if camera_id in active_cameras:
                    camera_data.update({
                        "url": active_cameras[camera_id].get("url"),
                        "location": active_cameras[camera_id].get("location", "Unknown"),
                        "type": active_cameras[camera_id].get("type", "unknown"),
                        "video_path": active_cameras[camera_id].get("video_path")
                    })
                
                camera = CameraRepository.create_camera(db, camera_data)
            
            # Load mask data
            with open(mask_path, 'r') as f:
                mask_data = json.load(f)
            
            # Create annotation
            annotation = AnnotationRepository.create_annotation(db, camera_id, mask_data)
            migrated_count += 1
        
        return {"message": f"Successfully migrated {migrated_count} mask files to database"}
    except Exception as e:
        logger.error(f"Error migrating data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error migrating data: {str(e)}")


# Add this after the existing routes
@app.post("/api/auto-generate-boxes")
async def auto_generate_boxes(
    image: UploadFile = File(...),
    confidence: float = Form(0.3)
):
    """
    Auto-generate parking spot boxes using the trained AI model
    """
    try:
        # Validate confidence threshold
        confidence = max(0.01, min(0.9, confidence))
        
        # Save uploaded image to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_image_path = temp_file.name
        
        try:
            # Initialize the AI model
            # Try to find a trained model
            model_paths = [ 
                "runs/detect/train3/weights/best.pt",
                "models/parking_spot_detector.pt",
                "dataset_augmented/runs/obb/train/weights/best.pt"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                # If no trained model found, create one with fallback detection
                logger.warning("No trained model found, using fallback detection")
                model = ParkingSpotAnnotator()
            else:
                logger.info(f"Using trained model: {model_path}")
                model = ParkingSpotAnnotator(model_path)
            
            # Detect parking spots
            spots_data = model.detect_parking_spots(
                image_path=temp_image_path, 
                conf=confidence
            )
            
            if not spots_data:
                return JSONResponse(
                    status_code=200,
                    content={
                        "boxes": [],
                        "groups": [],
                        "message": "No parking spots detected"
                    }
                )
            
            return JSONResponse(
                status_code=200,
                content={
                    "boxes": spots_data.get("boxes", []),
                    "groups": spots_data.get("groups", []),
                    "camera_id": spots_data.get("camera_id", "ai_generated"),
                    "confidence_threshold": confidence,
                    "message": f"Generated {len(spots_data.get('boxes', []))} parking spots"
                }
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                
    except Exception as e:
        logger.error(f"Error in auto_generate_boxes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Failed to generate boxes: {str(e)}",
                "boxes": [],
                "groups": []
            }
        )
        
        

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


