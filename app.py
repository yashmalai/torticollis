import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import math
from PIL import Image, ImageTk

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

class CervicalDystoniaAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор кривошеи")
        
        self.setup_ui()
        self.setup_mediapipe()
        self.setup_video_capture()
        
        self.is_running = False
        self.delay = 15

    def setup_ui(self):
        self.video_frame = ttk.LabelFrame(self.root, text="Видео с веб-камеры")
        self.video_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(padx=10, pady=5, fill=tk.X)
        
        self.btn_start = ttk.Button(self.control_frame, text="Старт", command=self.start_analysis)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        self.btn_stop = ttk.Button(self.control_frame, text="Стоп", command=self.stop_analysis)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        
        self.stats_frame = ttk.LabelFrame(self.root, text="Диагностические параметры")
        self.stats_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Создаем элементы для отображения всех 5 параметров
        self.stats_labels = {}
        parameters = [
            ("Ротация головы", "rotation"),
            ("Наклон к плечу", "tilt"),
            ("Наклон вперед/назад", "flexion"),
            ("Боковое смещение", "lateral_shift"),
            ("Продольное смещение", "longitudinal_shift")
        ]
        
        for row, (name, key) in enumerate(parameters):
            label = ttk.Label(self.stats_frame, text=name, width=25, anchor=tk.W)
            label.grid(row=row, column=0, padx=5, pady=2, sticky=tk.W)
            
            value_label = ttk.Label(self.stats_frame, text="N/A", width=40, anchor=tk.W)
            value_label.grid(row=row, column=1, padx=5, pady=2, sticky=tk.W)
            
            self.stats_labels[key] = value_label

    def setup_mediapipe(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def setup_video_capture(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Не удалось подключиться к веб-камере.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            print(f"Ошибка: {e}")
            self.cap = None

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
        
        # Обработка данных позы
        pose_results = self.pose.process(frame)
        if pose_results.pose_landmarks:
            results.update(self.calculate_pose_parameters(pose_results.pose_landmarks))
        
        # Обработка данных лица
        face_results = self.face_mesh.process(frame)
        if face_results.multi_face_landmarks and pose_results.pose_landmarks:
            results.update(self.calculate_head_flexion(
                pose_results.pose_landmarks,
                face_results.multi_face_landmarks[0]
            ))
        
        return frame, results

    def calculate_pose_parameters(self, landmarks):
        params = {}
        
        # Ротация головы
        left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]
        dx = right_ear.x - left_ear.x
        dy = right_ear.y - left_ear.y
        params['rotation'] = abs(math.degrees(math.atan2(dy, dx)) - 90)
        
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
        params['tilt'] = math.degrees(math.atan2(dy, dx))
        
        # Боковое смещение
        head_center_x = (left_ear.x + right_ear.x)/2
        params['lateral_shift'] = 1 if abs(head_center_x - shoulder_center[0]) > 0.05 else 0
        
        # Продольное смещение
        nose = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        params['longitudinal_shift'] = 1 if abs(nose.y - shoulder_center[1]) > 0.1 else 0
        
        return params

    def calculate_head_flexion(self, pose_landmarks, face_landmarks):
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        neck_base = (
            (left_shoulder.x + right_shoulder.x)/2,
            (left_shoulder.y + right_shoulder.y)/2
        )
        
        chin = face_landmarks.landmark[152]
        dx = chin.x - neck_base[0]
        dy = chin.y - neck_base[1]
        return {'flexion': math.degrees(math.atan2(dy, dx))}

    def classify_parameters(self, results):
        classifications = {}
        
        # Классификация ротации
        rotation = results.get('rotation', 0)
        if rotation < 1:
            classifications['rotation'] = "Отсутствует (0°)"
        elif 1 <= rotation <= 22:
            classifications['rotation'] = f"Незначительная ({int(rotation)}°)"
        elif 23 <= rotation <= 45:
            classifications['rotation'] = f"Легкая ({int(rotation)}°)"
        elif 46 <= rotation <= 67:
            classifications['rotation'] = f"Умеренная ({int(rotation)}°)"
        else:
            classifications['rotation'] = f"Выраженная ({int(rotation)}°)"
        
        # Классификация наклона к плечу
        tilt = abs(results.get('tilt', 0))
        if tilt < 1:
            classifications['tilt'] = "Отсутствует (0°)"
        elif 1 <= tilt <= 15:
            classifications['tilt'] = f"Легкий ({int(tilt)}°)"
        elif 16 <= tilt <= 35:
            classifications['tilt'] = f"Умеренный ({int(tilt)}°)"
        else:
            classifications['tilt'] = f"Выраженный ({int(tilt)}°)"
        
        # Классификация наклона вперед/назад
        flexion = results.get('flexion', 0)
        if flexion < -30:
            classifications['flexion'] = "Выраженный наклон назад"
        elif -30 <= flexion < -15:
            classifications['flexion'] = "Умеренный наклон назад"
        elif -15 <= flexion < -5:
            classifications['flexion'] = "Легкий наклон назад"
        elif -5 <= flexion <= 5:
            classifications['flexion'] = "Нормальное положение"
        elif 5 < flexion <= 20:
            classifications['flexion'] = "Легкий наклон вперед"
        elif 20 < flexion <= 35:
            classifications['flexion'] = "Умеренный наклон вперед"
        else:
            classifications['flexion'] = "Выраженный наклон вперед"
        
        # Классификация смещений
        classifications['lateral_shift'] = "Присутствует" if results.get('lateral_shift', 0) else "Отсутствует"
        classifications['longitudinal_shift'] = "Присутствует" if results.get('longitudinal_shift', 0) else "Отсутствует"
        
        return classifications

    def update_gui(self, frame, results):
        # Очистка холста
        self.canvas.delete("all")
        
        # Обновление изображения
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        # Обновление статистики
        if results:
            classifications = self.classify_parameters(results)
            for key, label in self.stats_labels.items():
                label.config(text=classifications.get(key, "N/A"))

    def on_closing(self):
        self.stop_analysis()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CervicalDystoniaAnalyzer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()