import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import math
from PIL import Image, ImageTk
import threading

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

class VideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("...")
        
        self.setup_ui()
        self.setup_mediapipe()
        self.setup_video_capture()
        
        self.is_running = False
        self.delay = 15  # Задержка между кадрами в ms

    def setup_ui(self):
        self.video_frame = ttk.LabelFrame(self.root, text="Video Feed")
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.btn_start = ttk.Button(self.control_frame, text="Start", command=self.start_analysis)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(self.control_frame, text="Stop", command=self.stop_analysis)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.stats_frame = ttk.LabelFrame(self.root, text="Real-time Statistics")
        self.stats_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.stats_labels = {}
        stats = [
            ("Head Rotation", "rotation"),
            ("Head Tilt", "tilt"),
            ("Flexion/Extension", "flexion"),
            ("Lateral Shift", "shift")
        ]
        
        for i, (name, key) in enumerate(stats):
            label = ttk.Label(self.stats_frame, text=f"{name}:")
            label.grid(row=i, column=0, padx=5, pady=2, sticky=tk.W)
            
            value_label = ttk.Label(self.stats_frame, text="N/A")
            value_label.grid(row=i, column=1, padx=5, pady=2, sticky=tk.W)
            
            self.stats_labels[key] = value_label

    def setup_mediapipe(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def setup_video_capture(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def start_analysis(self):
        if not self.is_running:
            self.is_running = True
            self.update_frame()

    def stop_analysis(self):
        self.is_running = False

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, results = self.process_frame(frame)
                self.update_gui(processed_frame, results)
            
            self.root.after(self.delay, self.update_frame)

    def process_frame(self, frame):
        results = {}
        
        # Обработка позы
        pose_results = self.pose.process(frame)
        if pose_results.pose_landmarks:
            results.update(self.calculate_angles(pose_results.pose_landmarks))
        
        # Обработка лица
        face_results = self.face_mesh.process(frame)
        if face_results.multi_face_landmarks:
            results.update(self.calculate_flexion(pose_results.pose_landmarks, face_results))
        
        # Визуализация
        annotated_frame = frame.copy()
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        
        return annotated_frame, results

    def calculate_angles(self, landmarks):
        results = {}
        
        # Ротация головы
        left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        dx = right_ear.x - left_ear.x
        dy = right_ear.y - left_ear.y
        results['rotation'] = abs(math.degrees(math.atan2(dy, dx)) - 90)
        
        # Наклон к плечу
        left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_center = (
            (left_shoulder.x + right_shoulder.x)/2,
            (left_shoulder.y + right_shoulder.y)/2
        )
        
        ear = left_ear if left_ear.y < right_ear.y else right_ear
        dx = ear.x - shoulder_center[0]
        dy = ear.y - shoulder_center[1]
        results['tilt'] = math.degrees(math.atan2(dy, dx))
        
        # Боковое смещение
        head_center = (left_ear.x + right_ear.x)/2
        results['shift'] = 1 if abs(head_center - shoulder_center[0]) > 0.05 else 0
        
        return results

    def calculate_flexion(self, pose_landmarks, face_results):
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        neck_base = (
            (left_shoulder.x + right_shoulder.x)/2,
            (left_shoulder.y + right_shoulder.y)/2
        )
        
        chin = face_results.multi_face_landmarks[0].landmark[152]
        dx = chin.x - neck_base[0]
        dy = chin.y - neck_base[1]
        return {'flexion': math.degrees(math.atan2(dy, dx))}

    def classify_condition(self, results):
        classification = []
        
        # Классификация ротации
        rotation = abs(results.get('rotation', 0))
        if rotation < 1: classification.append("Нет ротации")
        elif 1 <= rotation <= 22: classification.append("Незначительная ротация")
        elif 23 <= rotation <= 45: classification.append("Легкая ротация")
        elif 46 <= rotation <= 67: classification.append("Умеренная ротация")
        else: classification.append("Выраженная ротация")
        
        return classification

    def update_gui(self, frame, results):
        # Обновление изображения
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        # Обновление статистики
        if results:
            for key, label in self.stats_labels.items():
                value = results.get(key, 'N/A')
                if isinstance(value, float):
                    label.config(text=f"{value:.2f}°")
                else:
                    label.config(text=str(value))
            
            classification = self.classify_condition(results)
            # Можно добавить отображение классификации

    def on_closing(self):
        self.stop_analysis()
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyzerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()