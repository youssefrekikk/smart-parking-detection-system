
A FastAPI-based system for detecting available parking spots using computer vision and YOLOv8.


## Features

- Real-time parking spot detection using YOLOv8
- Support for multiple cameras (webcam, IP camera, video files)
- Interactive web interface for defining parking spots
- Grouping of parking spots by location
- PostgreSQL database for data persistence
- WebSocket streaming for real-time video

## Technologies Used

- FastAPI
- SQLAlchemy with PostgreSQL
- OpenCV
- YOLOv8 for object detection
- Fabric.js for UI interactions
- Alembic for database migrations
- WebSockets for real-time communication