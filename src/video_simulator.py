import cv2
import numpy as np
import asyncio
import threading
import time
from typing import Optional

class VideoSimulator:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.frame = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self):
        """Start the video simulation in a separate thread"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")
            
        self.is_running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        print(f"Video simulator thread started for {self.video_path}")

    async def start_async(self):
        """Start the video simulation asynchronously"""
        return self.start()

    def _update_frame(self):
        """Background thread function to continuously update the frame"""
        while self.is_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                # Reset to beginning of video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)  # Small delay before retrying
                    continue
            
            # Update the frame with thread safety
            with self.lock:
                self.frame = frame
            
            # Add a small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current frame from the video (synchronous version)"""
        if not self.is_running or self.cap is None:
            return None
            
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    async def get_frame_async(self) -> Optional[np.ndarray]:
        """Get the current frame from the video (asynchronous version)"""
        return self.get_frame()

    def stop(self):
        """Stop the video simulation"""
        self.is_running = False
        
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
        if self.cap is not None:
            self.cap.release()
            
        self.cap = None
        self.frame = None
        print(f"Video simulator stopped for {self.video_path}")