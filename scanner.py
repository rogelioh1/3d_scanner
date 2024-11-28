import sys
import os
import cv2
import freenect
import numpy as np
from playsound import playsound
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QWidget, QInputDialog
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
import face_recognition
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

def augment_image_pair(rgb_image, depth_image):
    augmented_pairs = []

    # Flip the image horizontally
    flipped_rgb = cv2.flip(rgb_image, 1)
    flipped_depth = cv2.flip(depth_image, 1)
    augmented_pairs.append((flipped_rgb, flipped_depth))

    # Rotate the image
    rows, cols, _ = rgb_image.shape
    for angle in [15, -15]:  # Rotate by 15 degrees
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_rgb = cv2.warpAffine(rgb_image, M, (cols, rows))
        rotated_depth = cv2.warpAffine(depth_image, M, (cols, rows))
        augmented_pairs.append((rotated_rgb, rotated_depth))

    # Adjust brightness for RGB only (depth remains unchanged)
    for factor in [0.7, 1.3]:  # Darker and brighter
        bright_rgb = np.clip(rgb_image * factor, 0, 255).astype(np.uint8)
        augmented_pairs.append((bright_rgb, depth_image))  # Keep depth as-is

    return augmented_pairs


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
            
class ScanThread(QThread):
    update_instruction = pyqtSignal(str)  # Emits the current instruction text
    update_distance = pyqtSignal(str)    # Emits the distance feedback message
    finished_scanning = pyqtSignal(str) # Emits the folder path upon scan completion

    def __init__(self, user_folder):
        super().__init__()
        self.user_folder = user_folder
        self.rgb_folder = os.path.join(self.user_folder, "rgb")
        self.depth_folder = os.path.join(self.user_folder, "depth")
        self.rgb_augmented_folder = os.path.join(self.user_folder, "rgb_augmented")
        self.depth_augmented_folder = os.path.join(self.user_folder, "depth_augmented")

        # Create folders
        os.makedirs(self.rgb_folder, exist_ok=True)
        os.makedirs(self.depth_folder, exist_ok=True)
        os.makedirs(self.rgb_augmented_folder, exist_ok=True)
        os.makedirs(self.depth_augmented_folder, exist_ok=True)

        self.instructions = [
            {"text": "Face forward", "audio": "audio/move.mp3"},
            {"text": "^^^^^ Look up ^^^^", "audio": "audio/move.mp3"},
            {"text": "\Look down/", "audio": "audio/move.mp3"},
            {"text": "<<<<<---- Look left", "audio": "audio/move.mp3"},
            {"text": "Look right ------>>>>>", "audio": "audio/move.mp3"}
        ]
        self.completion_sound = "audio/complete.mp3"
        self.instruction_index = 0
        self.max_frames_per_instruction = 60
        self.total_frame_count = 0
        self.distance_tolerance = 10
        self.depth_history = deque(maxlen=10)
        self.last_face_location = None
        self.sound_threads = []  # Keep track of all active SoundPlayer threads

    def play_audio(self, audio_file):
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return

        sound_thread = SoundPlayer(audio_file)
        self.sound_threads.append(sound_thread)  # Keep reference to prevent garbage collection
        sound_thread.finished.connect(lambda: self.sound_threads.remove(sound_thread))  # Remove thread when done
        sound_thread.start()

    def run(self):
        # Emit the first instruction and play its audio
        self.update_instruction.emit(self.instructions[self.instruction_index]["text"])
        self.play_audio(self.instructions[self.instruction_index]["audio"])
        print(f"Starting scan for user, saving data in {self.user_folder}")

        while self.instruction_index < len(self.instructions):
            rgb_frame = get_rgb_frame()
            depth_frame = get_depth_frame()

            if rgb_frame is None or depth_frame is None:
                print("Error: Could not get frames from Kinect.")
                continue

            rgb_small = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(rgb_small)

            if not face_locations:
                # Skip frames where no face is detected
                print("No face detected. Waiting...")
                QThread.msleep(100)  # Wait briefly before trying again
                continue

            # Scale face location back to original size
            self.last_face_location = [coord * 4 for coord in face_locations[0]]
            print(f"Face detected at: {self.last_face_location}")

            # Match depth frame resolution to RGB frame resolution
            depth_frame_resized = cv2.resize(depth_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))

            try:
                # Crop RGB and Depth using the same coordinates
                cropped_rgb = crop_face(rgb_frame, self.last_face_location)
                cropped_depth = crop_face(depth_frame_resized, self.last_face_location)

                # Save original images in their respective folders
                rgb_filename = os.path.join(self.rgb_folder, f"rgb_{self.total_frame_count:04d}.png")
                depth_filename = os.path.join(self.depth_folder, f"depth_{self.total_frame_count:04d}.png")
                cv2.imwrite(rgb_filename, cropped_rgb)
                cv2.imwrite(depth_filename, cropped_depth)
                print(f"Captured cropped frame {self.total_frame_count} using last known location")

                # Apply augmentations to RGB and depth images
                augmented_pairs = augment_image_pair(cropped_rgb, cropped_depth)
                for idx, (aug_rgb, aug_depth) in enumerate(augmented_pairs):
                    aug_rgb_filename = os.path.join(self.rgb_augmented_folder, f"rgb_aug_{self.total_frame_count:04d}_{idx}.png")
                    aug_depth_filename = os.path.join(self.depth_augmented_folder, f"depth_aug_{self.total_frame_count:04d}_{idx}.png")
                    cv2.imwrite(aug_rgb_filename, aug_rgb)
                    cv2.imwrite(aug_depth_filename, aug_depth)
                    print(f"Saved augmented pair: {aug_rgb_filename}, {aug_depth_filename}")

            except Exception as e:
                print(f"Error cropping using last known location: {e}")

            self.total_frame_count += 1

            # Move to the next instruction after capturing enough frames
            if self.total_frame_count % self.max_frames_per_instruction == 0:
                self.instruction_index += 1
                if self.instruction_index < len(self.instructions):
                    instruction = self.instructions[self.instruction_index]
                    print(f"Next instruction: {instruction['text']}")
                    self.update_instruction.emit(instruction["text"])
                    self.play_audio(instruction["audio"])

        print("Scan complete!")
        self.play_audio(self.completion_sound)
        print(f"Finished scanning for user. Data saved in {self.user_folder}")
        self.finished_scanning.emit(self.user_folder)


class KinectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Face Scanner")
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

        # Add instruction label
        self.instruction_label = QLabel("", self)
        self.instruction_label.setFont(QFont("Arial", 14))
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.instruction_label)

        # Add Begin Scan button
        self.begin_scan_button = QPushButton("Begin Scan", self)
        self.begin_scan_button.setFont(QFont("Arial", 14))
        self.begin_scan_button.setStyleSheet("background-color: #5cb85c; color: white;")
        self.begin_scan_button.setFixedWidth(200)
        self.begin_scan_button.setFixedHeight(50)
        self.begin_scan_button.clicked.connect(self.register_user)
        self.layout.addWidget(self.begin_scan_button, alignment=Qt.AlignCenter)

        # Placeholder for Kinect frames and state
        self.rgb_frame = None
        self.depth_frame = None
        self.scan_thread = None
        self.distance_message = ""

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

            # Draw a guide overlay on the mirrored frcame
            overlay = mirrored_frame.copy()
            height, width, _ = overlay.shape

            # Define colors (in BGR format)
            circle_color = (255, 255, 255)  # White circle
            rectangle_color = (0, 255, 0) # Green rectangle

            # Draw the last known crop rectangle if available
            if self.scan_thread and self.scan_thread.last_face_location:
                top, right, bottom, left = self.scan_thread.last_face_location
                # Mirror the crop rectangle for display
                left, right = width - right, width - left
                cv2.rectangle(overlay, (left, top), (right, bottom), rectangle_color, 2)

            # Draw a central guide circle
            cv2.circle(overlay, (width // 2, height // 2), 100, circle_color, 2)

            # Convert overlay to QImage for PyQt
            overlay_qimg = QImage(overlay.data, width, height, 3 * width, QImage.Format_RGB888)

            # Use QPainter on the QImage
            painter = QPainter(overlay_qimg)
            painter.setPen(QColor(0, 255, 0))  # green text
            painter.setFont(QFont("Arial", 16))
            painter.drawText(10, 30, "Try to keep your head inside both the circle and square.")
            painter.end()
            

            # Display the QImage in the GUI
            pixmap = QPixmap.fromImage(overlay_qimg)
            self.camera_label.setPixmap(pixmap)

    def register_user(self):
        # Prompt for user's name
        name, ok = QInputDialog.getText(self, "Register User", "Enter your name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        user_folder = os.path.join("scans", name)
        os.makedirs(user_folder, exist_ok=True)

        # Start scanning in a separate thread
        self.scan_thread = ScanThread(user_folder)
        self.scan_thread.update_instruction.connect(self.display_instruction)
        self.scan_thread.update_distance.connect(self.update_distance)
        self.scan_thread.finished_scanning.connect(self.on_scan_complete)
        self.scan_thread.start()

    def display_instruction(self, instruction):
        self.instruction_label.setText(instruction)

    def update_distance(self, message):
        self.distance_message = message

    def on_scan_complete(self, user_folder):
        print(f"Registration completed. Data saved in {user_folder}")
        self.instruction_label.setText("Scan complete")

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
