import sys
import os
import freenect
import numpy as np
from playsound import playsound
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QWidget, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
ci_build_and_not_headless = False
import cv2
from collections import deque

# Kinect distance thresholds (in depth sensor values)
MIN_DISTANCE = 100  # minimum acceptable distance
MAX_DISTANCE = 250  # maximum acceptable distance

# Function to capture RGB data from Kinect
def get_rgb_frame():
    frame, _ = freenect.sync_get_video()
    if frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return None

# Function to capture depth data from Kinect
def get_depth_frame():
    depth, _ = freenect.sync_get_depth()
    if depth is not None:
        return depth.astype(np.uint8)
    return None

# Function to crop a face and resize it to 256x256
def crop_face(image, face_location):
    top, right, bottom, left = face_location
    face = image[top:bottom, left:right]
    return cv2.resize(face, (256, 256))

class SoundPlayer(QThread):
    finished = pyqtSignal()

    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file

    def run(self):
        try:
            playsound(self.audio_file)
        except Exception as e:
            print(f"Error playing sound {self.audio_file}: {e}")
        finally:
            self.finished.emit()  # Signal that playback is done
            
# Import necessary libraries (unchanged from your original code)

# Modified ScanThread class
class ScanThread(QThread):
    update_instruction = pyqtSignal(str)
    update_distance = pyqtSignal(str)
    finished_scanning = pyqtSignal(str)

    def __init__(self, user_folder):
        super().__init__()
        self.user_folder = user_folder
        os.makedirs(self.user_folder, exist_ok=True)

        self.completion_sound = "audio/complete.mp3"
        self.max_frames = 200  # Limit the number of frames captured
        self.total_frame_count = 0
        self.depth_history = deque(maxlen=10)

    def run(self):
        print(f"Starting scan for objects, saving data in {self.user_folder}")

        while self.total_frame_count < self.max_frames:
            rgb_frame = get_rgb_frame()
            depth_frame = get_depth_frame()

            if rgb_frame is None or depth_frame is None:
                print("Error: Could not get frames from Kinect.")
                continue

            # Calculate center crop for 256x256
            height, width, _ = rgb_frame.shape
            center_x, center_y = width // 2, height // 2
            half_crop = 128  # Half of 256

            rgb_cropped = rgb_frame[
                center_y - half_crop:center_y + half_crop,
                center_x - half_crop:center_x + half_crop,
            ]
            depth_cropped = depth_frame[
                center_y - half_crop:center_y + half_crop,
                center_x - half_crop:center_x + half_crop,
            ]

            # Save cropped images
            rgb_filename = os.path.join(self.user_folder, f"rgb_{self.total_frame_count:04d}.png")
            depth_filename = os.path.join(self.user_folder, f"depth_{self.total_frame_count:04d}.png")
            cv2.imwrite(rgb_filename, rgb_cropped)
            cv2.imwrite(depth_filename, depth_cropped)

            print(f"Captured frame {self.total_frame_count}: RGB and Depth saved.")
            self.total_frame_count += 1

        print("Object scanning complete!")
        self.finished_scanning.emit(self.user_folder)

# Modified KinectApp class
class KinectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Scanner")
        self.setGeometry(100, 100, 800, 600)

        # Main widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Main layout
        self.layout = QVBoxLayout(self.central_widget)

        # Add camera feed
        self.camera_label = QLabel(self)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.camera_label)

        # Add Begin Scan button
        self.begin_scan_button = QPushButton("Begin Scan", self)
        self.begin_scan_button.setFont(QFont("Arial", 14))
        self.begin_scan_button.setStyleSheet("background-color: #5cb85c; color: white;")
        self.begin_scan_button.setFixedWidth(200)
        self.begin_scan_button.setFixedHeight(50)
        self.begin_scan_button.clicked.connect(self.register_scan)
        self.layout.addWidget(self.begin_scan_button, alignment=Qt.AlignCenter)

        # Placeholder for Kinect frames
        self.rgb_frame = None
        self.scan_thread = None

        # Timer for camera updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_view)
        self.timer.start(30)

    def update_camera_view(self):
        # Fetch RGB frame
        self.rgb_frame = get_rgb_frame()
        if self.rgb_frame is not None:
            # Mirror the frame for the preview
            mirrored_frame = cv2.flip(self.rgb_frame, 1)  # Flip horizontally

            # Draw a central guide rectangle
            overlay = mirrored_frame.copy()
            height, width, _ = overlay.shape
            center_x, center_y = width // 2, height // 2
            half_crop = 128  # Half of 256
            cv2.rectangle(
                overlay,
                (center_x - half_crop, center_y - half_crop),
                (center_x + half_crop, center_y + half_crop),
                (0, 255, 0), 2,
            )

            # Convert overlay to QImage for PyQt
            overlay_qimg = QImage(overlay.data, width, height, 3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(overlay_qimg)
            self.camera_label.setPixmap(pixmap)

    def register_scan(self):
        # Prompt for scan folder name
        name, ok = QInputDialog.getText(self, "Scan Object", "Enter object name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        object_folder = os.path.join("scans", name)
        os.makedirs(object_folder, exist_ok=True)

        # Start scanning in a separate thread
        self.scan_thread = ScanThread(object_folder)
        self.scan_thread.finished_scanning.connect(self.on_scan_complete)
        self.scan_thread.start()

    def on_scan_complete(self, folder_path):
        print(f"Scan completed. Data saved in {folder_path}")

    def closeEvent(self, event):
        # Cleanup on close
        self.timer.stop()
        if self.scan_thread and self.scan_thread.isRunning():
            self.scan_thread.terminate()
        cv2.destroyAllWindows()
        freenect.sync_stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = KinectApp()
    main_window.show()
    sys.exit(app.exec_())
