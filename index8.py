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
import json
import os
from datetime import datetime

# Constants
EAR_THRESHOLD = 0.2
ALERT_DURATION = 2  # seconds of closed eyes to trigger alert
EAR_HISTORY = deque(maxlen=10)  # Store last 10 EAR values

# Default Settings
DEFAULT_SETTINGS = {
    "sensitivity": 0.2,
    "alert_volume": 50,
    "alert_type": "صوت فقط",
    "camera_source": "الكاميرا الافتراضية",
    "resolution": "640x480",
    "report_interval": "كل ساعة",
    "auto_save_reports": True
}

# Load settings from file or use defaults
try:
    with open('settings.json', 'r', encoding='utf-8') as f:
        SETTINGS = json.load(f)
except FileNotFoundError:
    SETTINGS = DEFAULT_SETTINGS

# Mediapipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    EAR_HISTORY.append(ear)
    return sum(EAR_HISTORY) / len(EAR_HISTORY)

def save_settings(settings):
    with open('settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, ensure_ascii=False, indent=4)

def play_alert(is_drowsy, stop_event):
    pygame.mixer.init()
    alert_sound = pygame.mixer.Sound("alert_sound.mp3")
    alert_sound.set_volume(SETTINGS['alert_volume'] / 100)
    
    while not stop_event.is_set():
        if is_drowsy.value == 1:
            if not pygame.mixer.music.get_busy():  # Improved check for sound playing
                pygame.mixer.music.play(-1)  # Loop the alert sound
        else:
            pygame.mixer.music.stop()

def process_video(is_drowsy, frame_queue, stop_event):
    global closed_count, start_time

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Error: Could not open webcam.")

        # Set resolution based on settings
        width, height = map(int, SETTINGS['resolution'].split('x'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        eye_closed_start_time = None

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

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

                    if avg_ear < SETTINGS['sensitivity']:
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

            time.sleep(0.05)

    except Exception as e:
        print(f"Error in video processing: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def analyze_fatigue():
    # Function to handle the analysis of fatigue if needed
    if SETTINGS['auto_save_reports']:
        save_fatigue_report()

def save_fatigue_report():
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fatigue_detected": closed_count
    }
    if not os.path.exists('reports'):
        os.makedirs('reports')
    with open(f'reports/report_{datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

class DrowsinessDetectorApp:
    def __init__(self, root, frame_queue, is_drowsy, stop_event):
        self.root = root
        self.frame_queue = frame_queue
        self.is_drowsy = is_drowsy
        self.stop_event = stop_event
        self.monitoring_active = False
        
        # Configure main window
        self.root.title("Drive Safe - نظام مراقبة السائق")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1a1a1a")
        
        # Style configuration
        self.setup_styles()
        
        # Create UI components
        self.create_video_frame()
        self.create_control_panel()
        
        # Start GUI update
        self.update_gui()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Custom.TFrame", background="#1a1a1a")
        self.style.configure("Custom.TLabel", 
                           background="#1a1a1a", 
                           foreground="#ffffff", 
                           font=("Cairo", 12))
        self.style.configure("Custom.TButton", 
                           font=("Cairo", 11, "bold"),
                           padding=10)
        self.style.configure("Status.TLabel",
                           background="#1a1a1a",
                           foreground="#00ff00",
                           font=("Cairo", 14, "bold"))

    def create_video_frame(self):
        self.video_container = ttk.Frame(self.root, style="Custom.TFrame")
        self.video_container.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self):
        self.control_panel = ttk.Frame(self.root, style="Custom.TFrame")
        self.control_panel.pack(pady=10, padx=20, fill=tk.X)

        # Status indicator
        self.status_frame = ttk.Frame(self.control_panel, style="Custom.TFrame")
        self.status_frame.pack(fill=tk.X, pady=10)

        self.status_label = ttk.Label(self.status_frame, 
                                    text="حالة النظام: غير نشط", 
                                    style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Control buttons
        self.create_control_buttons()

    def create_control_buttons(self):
        self.button_frame = ttk.Frame(self.control_panel, style="Custom.TFrame")
        self.button_frame.pack(fill=tk.X, pady=10)

        buttons = [
            ("بدء المراقبة", self.toggle_monitoring, "#28a745"),
            ("تقرير التعب", self.show_fatigue_report, "#17a2b8"),
            ("الإعدادات", self.open_settings, "#6c757d"),
            ("إعادة ضبط", self.reset_settings, "#ffc107"),
            ("التعليمات", self.show_instructions, "#6610f2"),
            ("خروج", self.quit_app, "#dc3545")
        ]

        for text, command, color in buttons:
            btn = tk.Button(self.button_frame,
                          text=text,
                          command=command,
                          bg=color,
                          fg="white",
                          font=("Cairo", 11, "bold"),
                          relief="solid")
            btn.pack(side=tk.LEFT, padx=10, pady=5)

    def update_gui(self):
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image = ImageTk.PhotoImage(image)
            self.video_label.configure(image=image)
            self.video_label.image = image

        self.root.after(20, self.update_gui)

    def toggle_monitoring(self):
        if self.monitoring_active:
            self.stop_event.set()
            self.status_label.config(text="حالة النظام: غير نشط", style="Status.TLabel")
        else:
            self.monitoring_active = True
            self.stop_event.clear()
            self.status_label.config(text="حالة النظام: نشط", style="Status.TLabel")
            p = Process(target=process_video, args=(self.is_drowsy, self.frame_queue, self.stop_event))
            p.start()
            alert_process = Process(target=play_alert, args=(self.is_drowsy, self.stop_event))
            alert_process.start()

    def show_fatigue_report(self):
        report_files = [f for f in os.listdir('reports') if f.endswith('.json')]
        if report_files:
            reports_text = "\n".join(report_files)
        else:
            reports_text = "لا توجد تقارير حالياً."
        messagebox.showinfo("تقارير التعب", reports_text)

    def open_settings(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("الإعدادات")
        settings_window.geometry("400x400")

        sensitivity_label = ttk.Label(settings_window, text="حساسية العين المغلقة:", font=("Cairo", 12))
        sensitivity_label.pack(pady=10)
        sensitivity_entry = ttk.Entry(settings_window, font=("Cairo", 12))
        sensitivity_entry.insert(0, str(SETTINGS['sensitivity']))
        sensitivity_entry.pack(pady=10)

        save_button = ttk.Button(settings_window, text="حفظ التعديلات", command=lambda: self.save_settings(settings_window, sensitivity_entry))
        save_button.pack(pady=20)

    def save_settings(self, window, sensitivity_entry):
        new_sensitivity = float(sensitivity_entry.get())
        SETTINGS['sensitivity'] = new_sensitivity
        save_settings(SETTINGS)
        window.destroy()

    def reset_settings(self):
        global SETTINGS
        SETTINGS = DEFAULT_SETTINGS
        save_settings(SETTINGS)
        messagebox.showinfo("إعادة ضبط", "تم إعادة الضبط إلى الإعدادات الافتراضية.")

    def show_instructions(self):
        instructions = (
            "هذا النظام يستخدم لتحذير السائق عند حدوث تعب.\n"
            "يتم الكشف عن تعبيرات الوجه مثل إغلاق العين.\n"
            "إذا تم اكتشاف تعب، سيتم تشغيل الصوت للتنبيه."
        )
        messagebox.showinfo("التعليمات", instructions)

    def quit_app(self):
        self.stop_event.set()
        self.root.quit()

def main():
    root = tk.Tk()
    frame_queue = Queue(maxsize=10)
    is_drowsy = Value('i', 0)
    stop_event = Event()
    
    app = DrowsinessDetectorApp(root, frame_queue, is_drowsy, stop_event)
    root.mainloop()

if __name__ == "__main__":
    main()
