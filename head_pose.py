"""
https://github.com/c0nn0r/AttentiveLLM
Connor Dickie August 2025
"""

import numpy as np
import cv2
import mediapipe as mp
import time
import socket
import threading
from enum import Enum

# Head pose direction enum
class HeadPoseDirection(Enum):
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

class HeadPoseDetector:
    def __init__(self, tcp_host="localhost", tcp_port=5000):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # TCP communication
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.socket = None
        self.last_forward_state = False
        self.connection_lock = threading.Lock()
        
        # Head pose thresholds (in degrees) - adjusted for better balance
        self.threshold = 8   # General threshold for up/down
        self.left_threshold = 6   # Lower threshold for left detection
        self.right_threshold = 6  # Lower threshold for right detection
        
        # Calibration data
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibration_offset = 0
        
        # Debug mode
        self.debug_mode = False
        
        # Initialize TCP connection
        self._init_tcp_connection()
    
    def _init_tcp_connection(self):
        """Initialize TCP connection to button2.py client"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.tcp_host, self.tcp_port))
            print(f"Connected to TCP server at {self.tcp_host}:{self.tcp_port}")
        except Exception as e:
            print(f"Failed to connect to TCP server: {e}")
            self.socket = None
    
    def _send_tcp_command(self, command):
        """Send TCP command in a non-blocking way"""
        if self.socket is None:
            return
        
        def send_command():
            try:
                with self.connection_lock:
                    self.socket.sendall(f"{command}\n".encode())
            except Exception as e:
                print(f"Failed to send TCP command: {e}")
                self.socket = None
        
        # Send in a separate thread to avoid blocking
        threading.Thread(target=send_command, daemon=True).start()
    
    def calibrate_forward_position(self, y_angle):
        """Calibrate the forward position by collecting samples"""
        if len(self.calibration_samples) < 30:  # Collect 30 samples
            self.calibration_samples.append(y_angle)
            return False
        
        if not self.is_calibrated:
            # Calculate the average forward position
            self.calibration_offset = np.mean(self.calibration_samples)
            self.is_calibrated = True
            print(f"Calibration complete. Forward offset: {self.calibration_offset:.2f} degrees")
        
        return True
    
    def determine_head_pose(self, x, y, z):
        """Determine head pose direction based on rotation angles with improved accuracy"""
        # Apply calibration offset
        y_adjusted = y - self.calibration_offset
        
        # Check yaw (left/right) first with asymmetric thresholds
        if y_adjusted <= -self.left_threshold:
            return HeadPoseDirection.LEFT
        elif y_adjusted >= self.right_threshold:
            return HeadPoseDirection.RIGHT
        # Check pitch (up/down) if not looking left/right
        elif x < -self.threshold:
            return HeadPoseDirection.DOWN
        elif x > self.threshold:
            return HeadPoseDirection.UP
        # If no significant rotation, looking forward
        else:
            return HeadPoseDirection.FORWARD
    
    def get_landmark_color(self, direction):
        """Get landmark color based on head pose direction"""
        if direction == HeadPoseDirection.FORWARD:
            return (0, 255, 0)  # Green for looking at camera
        else:
            return (0, 0, 255)  # Red for not looking at camera
    
    def process_frame(self, image):
        """Process a single frame and return processed image and head pose info"""
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = self.face_mesh.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, img_c = image.shape
        head_pose_direction = None
        is_looking_at_camera = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_2d = []
                face_3d = []
                
                # Enhanced landmark selection for better head pose estimation
                # Using more landmarks for better accuracy
                landmark_indices = [
                    33, 263, 1, 61, 291, 199,  # Original landmarks
                    10, 338, 297, 332, 284, 251,  # Additional landmarks for better accuracy
                    389, 356, 454, 323, 361, 288,  # More facial landmarks
                    397, 365, 379, 378, 400, 377,  # Cheek and jaw landmarks
                    152, 148, 176, 149, 150, 136,  # Eye and nose landmarks
                    21, 54, 103, 67, 109, 10      # Additional reference points
                ]
                
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in landmark_indices:
                        if idx == 1:  # Nose tip
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                
                if len(face_2d) >= 6:  # Ensure we have enough landmarks
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    
                    # Improved camera matrix with better focal length estimation
                    focal_length = max(img_w, img_h)  # Use the larger dimension
                    cam_matrix = np.array([
                        [focal_length, 0, img_w/2],
                        [0, focal_length, img_h/2],
                        [0, 0, 1]
                    ])
                    distortion_matrix = np.zeros((4, 1), dtype=np.float64)
                    
                    # Solve PnP to get rotation and translation
                    success, rotation_vec, translation_vec = cv2.solvePnP(
                        face_3d, face_2d, cam_matrix, distortion_matrix,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        # Get rotation matrix
                        rmat, jac = cv2.Rodrigues(rotation_vec)
                        
                        # Decompose rotation matrix to get Euler angles
                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                        
                        x = angles[0] * 360  # Pitch (up/down)
                        y = angles[1] * 360  # Yaw (left/right)
                        z = angles[2] * 360  # Roll (tilt)
                        
                        # Calibrate if not done yet
                        if not self.is_calibrated:
                            self.calibrate_forward_position(y)
                        
                        # Debug mode
                        if self.debug_mode:
                            self.debug_current_values(x, y, z)
                            self.debug_mode = False
                        
                        # Determine head pose direction
                        head_pose_direction = self.determine_head_pose(x, y, z)
                        is_looking_at_camera = (head_pose_direction == HeadPoseDirection.FORWARD)
                        
                        # Handle TCP communication for looking at camera
                        if is_looking_at_camera and not self.last_forward_state:
                            self._send_tcp_command("play")
                        elif not is_looking_at_camera and self.last_forward_state:
                            self._send_tcp_command("pause")
                        
                        self.last_forward_state = is_looking_at_camera
                        
                        # Draw head pose indicator
                        nose_3d_projection, jacobian = cv2.projectPoints(
                            nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix
                        )
                        
                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                        
                        # Draw direction line
                        cv2.line(image, p1, p2, (255, 0, 0), 3)
                        
                        # Display head pose information with calibration data
                        text = head_pose_direction.value.upper()
                        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                        
                        # Show raw and adjusted angles
                        y_adjusted = y - self.calibration_offset if self.is_calibrated else y
                        cv2.putText(image, f"x: {np.round(x, 2)}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image, f"y: {np.round(y, 2)}", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(image, f"y_adj: {np.round(y_adjusted, 2)}", (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.putText(image, f"z: {np.round(z, 2)}", (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # Show calibration status
                        cal_status = "Calibrated" if self.is_calibrated else f"Calibrating... ({len(self.calibration_samples)}/30)"
                        cv2.putText(image, cal_status, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Show thresholds and current detection info
                        cv2.putText(image, f"L:{self.left_threshold} R:{self.right_threshold}", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(image, f"y_adj: {np.round(y_adjusted, 2)}", (20, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(image, f"Direction: {head_pose_direction.value}", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                        # Draw facial landmarks with appropriate color
                        landmark_color = self.get_landmark_color(head_pose_direction)
                        drawing_spec = self.mp_drawing.DrawingSpec(
                            color=landmark_color, thickness=2, circle_radius=1
                        )
                        
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec
                        )
        
        return image, head_pose_direction, is_looking_at_camera

    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration_samples = []
        self.is_calibrated = False
        self.calibration_offset = 0
        print("Calibration reset")
    
    def adjust_thresholds(self, left_threshold=None, right_threshold=None, general_threshold=None):
        """Adjust detection thresholds"""
        if left_threshold is not None:
            self.left_threshold = left_threshold
            print(f"Left threshold set to: {left_threshold}")
        if right_threshold is not None:
            self.right_threshold = right_threshold
            print(f"Right threshold set to: {right_threshold}")
        if general_threshold is not None:
            self.threshold = general_threshold
            print(f"General threshold set to: {general_threshold}")
        print(f"Current thresholds - L:{self.left_threshold} R:{self.right_threshold} G:{self.threshold}")

    def test_detection_logic(self):
        """Test the detection logic with sample values"""
        print("\n=== Testing Detection Logic ===")
        print(f"Current thresholds - L:{self.left_threshold} R:{self.right_threshold} G:{self.threshold}")
        
        test_cases = [
            (-15, "LEFT"),   # Should be LEFT
            (-10, "LEFT"),   # Should be LEFT  
            (-8, "LEFT"),    # Should be LEFT (at threshold)
            (-6, "LEFT"),    # Should be LEFT (at threshold)
            (-5, "FORWARD"), # Should be FORWARD
            (0, "FORWARD"),  # Should be FORWARD
            (5, "FORWARD"),  # Should be FORWARD
            (6, "RIGHT"),    # Should be RIGHT (at threshold)
            (8, "RIGHT"),    # Should be RIGHT (at threshold)
            (10, "RIGHT"),   # Should be RIGHT
            (15, "RIGHT"),   # Should be RIGHT
        ]
        
        for y_val, expected in test_cases:
            # Test without calibration offset
            y_adjusted = y_val  # Don't apply calibration offset for testing
            result = None
            
            # Apply the same logic as determine_head_pose but without calibration
            if y_adjusted <= -self.left_threshold:
                result = HeadPoseDirection.LEFT
            elif y_adjusted >= self.right_threshold:
                result = HeadPoseDirection.RIGHT
            elif 0 < -self.threshold:  # This will never be true for x=0
                result = HeadPoseDirection.DOWN
            elif 0 > self.threshold:   # This will never be true for x=0
                result = HeadPoseDirection.UP
            else:
                result = HeadPoseDirection.FORWARD
            
            status = "✓" if result.value.upper() == expected else "✗"
            print(f"{status} y={y_val:3d} -> {result.value.upper():8s} (expected: {expected}) [L_thresh={-self.left_threshold}, R_thresh={self.right_threshold}]")
        print("=== End Test ===\n")

    def debug_current_values(self, x, y, z):
        """Debug function to show current values and thresholds"""
        y_adjusted = y - self.calibration_offset
        print(f"DEBUG: x={x:.2f}, y={y:.2f}, y_adj={y_adjusted:.2f}")
        print(f"DEBUG: L_thresh={-self.left_threshold}, R_thresh={self.right_threshold}")
        print(f"DEBUG: y_adj <= -L_thresh: {y_adjusted <= -self.left_threshold}")
        print(f"DEBUG: y_adj >= R_thresh: {y_adjusted >= self.right_threshold}")
        
        # Test the logic
        if y_adjusted <= -self.left_threshold:
            print("DEBUG: Would detect LEFT")
        elif y_adjusted >= self.right_threshold:
            print("DEBUG: Would detect RIGHT")
        else:
            print("DEBUG: Would detect FORWARD")
        print("---")

def main():
    # Initialize head pose detector
    detector = HeadPoseDetector()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    print("Head Pose Detection Started")
    print("Press 'ESC' to exit")
    print("Press 'R' to reset calibration")
    print("Press '1'/'2' to decrease/increase left threshold")
    print("Press '3'/'4' to decrease/increase right threshold")
    print("Press '5'/'6' to decrease/increase general threshold")
    print("Press 'T' to test detection logic")
    print("Press 'D' to debug current values")
    
    while cap.isOpened():
        success, image = cap.read()
        
        if not success:
            print("Error: Could not read frame")
            break
        
        start = time.time()
        
        # Process frame
        processed_image, head_pose, is_looking_at_camera = detector.process_frame(image)
        
        # Calculate and display FPS
        end = time.time()
        total_time = end - start
        fps = 1 / total_time if total_time > 0 else 0
        
        cv2.putText(processed_image, f'FPS: {int(fps)}', (20, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        # Display status
        status_text = "Looking at Camera" if is_looking_at_camera else "Not Looking at Camera"
        status_color = (0, 255, 0) if is_looking_at_camera else (0, 0, 255)
        cv2.putText(processed_image, status_text, (20, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Show the processed image
        cv2.imshow('Head Pose Detection', processed_image)
        
        # Handle keyboard input
        key = cv2.waitKey(5) & 0xFF
        if key == 27:  # ESC key
            break
        elif key == ord('r') or key == ord('R'):  # Reset calibration
            detector.reset_calibration()
        elif key == ord('1'):  # Decrease left threshold
            detector.adjust_thresholds(left_threshold=max(1, detector.left_threshold - 1))
        elif key == ord('2'):  # Increase left threshold
            detector.adjust_thresholds(left_threshold=detector.left_threshold + 1)
        elif key == ord('3'):  # Decrease right threshold
            detector.adjust_thresholds(right_threshold=max(1, detector.right_threshold - 1))
        elif key == ord('4'):  # Increase right threshold
            detector.adjust_thresholds(right_threshold=detector.right_threshold + 1)
        elif key == ord('5'):  # Decrease general threshold
            detector.adjust_thresholds(general_threshold=max(1, detector.threshold - 1))
        elif key == ord('6'):  # Increase general threshold
            detector.adjust_thresholds(general_threshold=detector.threshold + 1)
        elif key == ord('t') or key == ord('T'):  # Test detection logic
            detector.test_detection_logic()
        elif key == ord('d') or key == ord('D'):  # Debug current values
            # We need to get current values from the last processed frame
            # This will be called in the next frame processing
            detector.debug_mode = True
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if detector.socket:
        detector.socket.close()

if __name__ == "__main__":
    main()

