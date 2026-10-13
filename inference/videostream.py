import cv2
import threading
import time
from typing import Optional, Union
import numpy as np


class VideoStream:
    """
    Simulates a live streaming camera by reading a video file frame-by-frame 
    in a background thread paced strictly at the video's native FPS.
    """

    def __init__(self, video_path: str, loop: bool = False, debug: bool = False):
        """
        :param video_path: Path to the video file.
        :param loop: If True, loops the video indefinitely like an ongoing live stream.
        :param debug: If True, processes frames as fast as possible with no FPS sleep delays.
        """
        self.video_path = video_path
        self.loop = loop
        self.debug = debug

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")

        # Read native FPS and calculate delay per frame
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or np.isnan(self.fps):
            self.fps = 30.0  # fallback if video metadata missing
        self.frame_delay = 1.0 / self.fps

        # Frame buffer and synchronization
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.is_stopped = False
        self.thread: Optional[threading.Thread] = None

        # Read initial frame to ensure buffer is not empty (for stream mode)
        if not self.debug:
            ret, first_frame = self.cap.read()
            if ret:
                self.frame = first_frame
            else:
                raise RuntimeError(f"Could not read first frame from {self.video_path}")

        # Start the streaming thread if not in synchronous debug mode
        self.start()

    def start(self) -> "VideoStream":
        """Starts the background reading thread (only in live streaming mode)."""
        if not self.debug and not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
        elif self.debug:
            self.running = True
        return self

    def _update_loop(self):
        """Thread loop that reads frames paced at video FPS (live streaming mode)."""
        while self.running:
            start_time = time.perf_counter()

            ret, new_frame = self.cap.read()

            if not ret:
                if self.loop:
                    # Rewind video to start
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # End of video reached
                    self.is_stopped = True
                    break

            with self.lock:
                self.frame = new_frame

            # Precise timing control to pace the stream to video FPS
            elapsed = time.perf_counter() - start_time
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Returns the next frame.
        - In debug mode: synchronously reads the next frame directly from cv2.VideoCapture (no skipped frames).
        - In streaming mode: returns the latest buffered frame from the background thread (thread-safe).
        """
        if self.debug:
            ret, frame = self.cap.read()
            if not ret:
                self.is_stopped = True
                return None
            return frame

        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def get_fps(self) -> float:
        """Returns the native FPS of the video."""
        return self.fps

    def is_alive(self) -> bool:
        """Returns True if the stream is actively reading frames."""
        return self.running and not self.is_stopped

    def stop(self):
        """Stops the background thread and releases resources."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
