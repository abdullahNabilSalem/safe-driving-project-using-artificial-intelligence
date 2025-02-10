import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
from multiprocessing import Process, Value, Queue, Event
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time
import colorsys
from collections import deque

# Constants
EAR_THRESHOLD = 0.2
ALERT_DURATION = 2  # seconds of closed eyes to trigger alert
EAR_HISTORY = deque(maxlen=10)  # Store last 10 EAR values

# Mediapipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR Calculation Function
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    EAR_HISTORY.append(ear)
    return sum(EAR_HISTORY) / len(EAR_HISTORY)  # Moving average

# Audio Playback Function
def play_alert(is_drowsy, stop_event):
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound("alert_sound.mp3")
    while not stop_event.is_set():
        if is_drowsy.value == 1:
            if not pygame.mixer.get_busy():
                alert_sound.play()
        else:
            pygame.mixer.stop()

# Video Processing Function
closed_count = 0
start_time = time.time()

def process_video(is_drowsy, frame_queue, stop_event):
    global closed_count, start_time

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Error: Could not open webcam.")

        eye_closed_start_time = None

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            # Reduce frame size for better performance
            frame = cv2.resize(frame, (700, 550))

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                 int(face_landmarks.landmark[i].y * frame.shape[0])) for i in LEFT_EYE]
                    right_eye = [(int(face_landmarks.landmark[i].x * frame.shape[1]), 
                                  int(face_landmarks.landmark[i].y * frame.shape[0])) for i in RIGHT_EYE]

                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0

                    if avg_ear < EAR_THRESHOLD:
                        if eye_closed_start_time is None:
                            eye_closed_start_time = time.time()
                        elif time.time() - eye_closed_start_time >= ALERT_DURATION:
                            is_drowsy.value = 1
                            closed_count += 1
                    else:
                        eye_closed_start_time = None
                        is_drowsy.value = 0

                    cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
                    cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)

            if time.time() - start_time >= 60:
                analyze_fatigue()
                start_time = time.time()

            if not frame_queue.full():
                frame_queue.put(frame)

            time.sleep(0.05)  # Reduce delay between frames

    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def analyze_fatigue():
    global closed_count, start_time
    fatigue_level = ""

    if closed_count < 5:
        fatigue_level = "Low"
    elif 5 <= closed_count <= 10:
        fatigue_level = "Medium"
    else:
        fatigue_level = "High"

    messagebox.showinfo("Fatigue Level", f"Fatigue Level: {fatigue_level}")

    if fatigue_level == "High":
        messagebox.showwarning("Warning", "You need to rest now!")
        pygame.mixer.Sound("alert_sound.mp3").play()

    closed_count = 0

class DrowsinessDetectorApp:
    def __init__(self, root, frame_queue, is_drowsy, stop_event):
        self.root = root
        self.frame_queue = frame_queue
        self.is_drowsy = is_drowsy
        self.stop_event = stop_event
        self.monitoring_active = False
        
        # Configure main window
        self.root.title("Drive Safe - Driver Monitoring System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        
        # Customize style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Custom.TFrame", background="#1a1a1a")
        self.style.configure("Custom.TLabel", 
                           background="#1a1a1a", 
                           foreground="#ffffff", 
                           font=("Cairo", 12))
        self.style.configure("Custom.TButton", 
                           font=("Cairo", 11, "bold"),
                           padding=10,
                           background="#007bff",
                           foreground="white")
        self.style.configure("Status.TLabel",
                           background="#1a1a1a",
                           foreground="#00ff00",
                           font=("Cairo", 14, "bold"))

        # Video frame
        self.video_container = ttk.Frame(self.root, style="Custom.TFrame")
        self.video_container.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Control panel
        self.control_panel = ttk.Frame(self.root, style="Custom.TFrame")
        self.control_panel.pack(pady=10, padx=20, fill=tk.X)

        # Status indicator
        self.status_frame = ttk.Frame(self.control_panel, style="Custom.TFrame")
        self.status_frame.pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(self.status_frame, 
                                    text="System Status: Inactive", 
                                    style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Sensitivity control
        self.threshold_frame = ttk.Frame(self.control_panel, style="Custom.TFrame")
        self.threshold_frame.pack(fill=tk.X, pady=10)

        self.threshold_label = ttk.Label(self.threshold_frame, 
                                       text="Sensitivity Level",
                                       style="Custom.TLabel")
        self.threshold_label.pack(side=tk.RIGHT, padx=10)

        self.threshold_scale = ttk.Scale(self.threshold_frame,
                                       from_=0.1,
                                       to=0.3,
                                       orient=tk.HORIZONTAL,
                                       value=EAR_THRESHOLD,
                                       command=self.update_threshold)
        self.threshold_scale.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)

        # Control buttons
        self.button_frame = ttk.Frame(self.control_panel, style="Custom.TFrame")
        self.button_frame.pack(fill=tk.X, pady=10)

        buttons = [
            ("Start Monitoring", self.toggle_monitoring, "#28a745"),
            ("Fatigue Report", self.show_fatigue_report, "#17a2b8"),
            ("Settings", self.open_settings, "#6c757d"),
            ("Reset", self.reset_threshold, "#ffc107"),
            ("Instructions", self.show_instructions, "#6610f2"),
            ("Exit", self.quit_app, "#dc3545")
        ]

        for text, command, color in buttons:
            btn = tk.Button(self.button_frame,
                          text=text,
                          command=command,
                          bg=color,
                          fg="white",
                          font=("Cairo", 11, "bold"),
                          relief=tk.FLAT,
                          padx=20,
                          pady=10)
            btn.pack(side=tk.RIGHT, padx=5)
            btn.bind("<Enter>", lambda e, btn=btn: btn.configure(bg=self.adjust_color(btn["bg"], -20)))
            btn.bind("<Leave>", lambda e, btn=btn, color=color: btn.configure(bg=color))

        # Start GUI update
        self.update_gui()

    def adjust_color(self, color, amount):
        """Adjust button color on hover"""
        try:
            # Convert color from Hex to RGB
            c = tuple(int(color.lstrip('#')[i:i+2], 16)/255. for i in (0, 2, 4))
            # Convert to HLS
            c = colorsys.rgb_to_hls(*c)
            # Adjust lightness
            adjusted = colorsys.hls_to_rgb(c[0], max(0, min(1, c[1] + amount/255.)), c[2])
            # Convert back to Hex
            return f"#{int(adjusted[0]*255):02x}{int(adjusted[1]*255):02x}{int(adjusted[2]*255):02x}"
        except:
            return color

    def update_threshold(self, value):
        global EAR_THRESHOLD
        EAR_THRESHOLD = float(value)
        self.threshold_label.config(text=f"Sensitivity Level: {EAR_THRESHOLD:.2f}")

    def reset_threshold(self):
        global EAR_THRESHOLD
        EAR_THRESHOLD = 0.2
        self.threshold_scale.set(EAR_THRESHOLD)
        self.is_drowsy.value = 0
        pygame.mixer.stop()
        messagebox.showinfo("Reset", "Settings have been reset to default values")

    def toggle_monitoring(self):
        self.monitoring_active = not self.monitoring_active
        if self.monitoring_active:
            self.status_label.config(text="System Status: Active", foreground="#00ff00")
        else:
            self.status_label.config(text="System Status: Inactive", foreground="#ff0000")

    def show_fatigue_report(self):
        global closed_count
        report_text = f"""
        Current Fatigue Report:
        
        Eye Closures: {closed_count}
        Current Sensitivity Level: {EAR_THRESHOLD:.2f}
        Elapsed Time: {int(time.time() - start_time)} seconds
        """
        messagebox.showinfo("Fatigue Report", report_text)

    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg="#1a1a1a")
        
        ttk.Label(settings_window, 
                 text="System Settings",
                 style="Custom.TLabel").pack(pady=20)
        
        # Add more settings here

    def update_gui(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            if self.monitoring_active:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            else:
                self.video_label.configure(image='')
                self.video_label.configure(text="Monitoring Paused", fg="white")

        self.root.after(10, self.update_gui)

    def quit_app(self):
        if messagebox.askokcancel("Confirm Exit", "Are you sure you want to exit?"):
            self.stop_event.set()  # Stop processes safely
            pygame.mixer.quit()
            self.root.destroy()

    def show_instructions(self):
        instructions_text = """
        Driver Monitoring System Instructions:

        1. Ensure the camera is properly positioned towards your face.
        2. Adjust the system sensitivity as needed.
        3. Click 'Start Monitoring' to activate the system.
        4. The system will alert you if drowsiness is detected.
        5. You can review the fatigue report at any time.
        6. Use the 'Reset' button to restore default settings.
        """
        messagebox.showinfo("Instructions", instructions_text)

# Main Function
if __name__ == "__main__":
    is_drowsy = Value('i', 0)
    frame_queue = Queue(maxsize=10)
    stop_event = Event()

    video_process = Process(target=process_video, args=(is_drowsy, frame_queue, stop_event))
    alert_process = Process(target=play_alert, args=(is_drowsy, stop_event))

    video_process.start()
    alert_process.start()

    root = tk.Tk()
    app = DrowsinessDetectorApp(root, frame_queue, is_drowsy, stop_event)
    root.mainloop()

    stop_event.set()  # Safely stop processes
    video_process.join()
    alert_process.join()