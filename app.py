import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import math
import numpy as np
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
        self.ref_x = None
        self.reference_displacement = None

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

    def calculate_angle(self, v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_theta = dot_product / (norm_v1 * norm_v2)
        theta = math.acos(np.clip(cos_theta, -1.0, 1.0))  # Угол в радианах
        return math.degrees(theta)  # Переводим в градусы

    def classify_head_tilt(self, angle, direction):
        if direction == "Наклон вперед":
            if angle < 10:
                return 0, "No"
            elif 10 <= angle < 15:
                return 1, "Simple"
            elif 15 <= angle < 25:
                return 2, "Have"
            else:
                return 3, "Extra"
        elif direction == "Наклон назад":
            if angle < 15:
                return 0, "No"
            elif 10 <= angle < 15:
                return 1, "Simple"
            elif 15 <= angle < 25:
                return 2, "Have"
            else:
                return 3, "Extra"

    def calculate_forward_displacement(self, face_landmarks, pose_landmarks, img_shape):
        # Опорные точки: кончик носа (1), середина плеч (11, 12)
        nose_tip = np.array([face_landmarks[1].x * img_shape[1], face_landmarks[1].y * img_shape[0]])
        left_shoulder = np.array([pose_landmarks[11].x * img_shape[1], pose_landmarks[11].y * img_shape[0]])
        right_shoulder = np.array([pose_landmarks[12].x * img_shape[1], pose_landmarks[12].y * img_shape[0]])

        # Средняя точка между плечами
        mid_shoulder = (left_shoulder + right_shoulder) / 2

        # Эвристика: расстояние от носа до линии плеч
        displacement = np.linalg.norm(nose_tip - mid_shoulder)
        return displacement
    
    def process_frame(self, frame):
        results = {}
        height, width, _ = frame.shape
        face_results = self.face_mesh.process(frame)
        pose_results = self.pose.process(frame)
        FACE_CENTER_LANDMARKS = [1, 4, 6, 9, 152]

        if face_results.multi_face_landmarks and pose_results.pose_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark # для лица
                pose_landmarks = pose_results.pose_landmarks.landmark # для тела

                # Наклон головы к плечу
                left_eye = (landmarks[33].x, landmarks[33].y)
                right_eye = (landmarks[263].x, landmarks[263].y)

                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                tilt = math.degrees(math.atan2(dy, dx))
                if tilt >= -10 or tilt <= 10:
                    head_direction = "Нормальное положение"
                elif tilt < -10:
                    head_direction = "Наклон вправо"
                else:
                    head_direction = "Наклон влево"

                results['tilt'] = f"{tilt:.2f} {head_direction}"

                # Наклон головы вперед или назад
                nose_tip = (int(landmarks[1].x * width), int(landmarks[1].y * height))  # Кончик носа
                forehead = (int(landmarks[10].x * width), int(landmarks[10].y * height))  # Лоб
                chin = (int(landmarks[152].x * width), int(landmarks[152].y * height))  # Подбородок

                face_vector = np.array([chin[0] - forehead[0], chin[1] - forehead[1]])
                vertical_vector = np.array([0, 1])

                angle = self.calculate_angle(face_vector, vertical_vector)

                tilt_direction = "Наклон вперед" if face_vector[0] > 0 else "Наклон назад"

                severity, description = self.classify_head_tilt(angle, tilt_direction)
                results['flexion'] = f"{angle:.2f} {tilt_direction} - Severity: {severity} ({description})"


                #Боковое смещение головы вправо или влево
                x_coords = [landmarks[i].x * width for i in FACE_CENTER_LANDMARKS]
                avg_x = sum(x_coords) / len(x_coords)  # Средняя координата X

                if self.ref_x is None:
                    self.ref_x = avg_x

                shift = avg_x - self.ref_x
                direction = 1 if shift < 200 or shift > 230 else 0
                results['lateral_shift'] = f"{direction} ({shift:.2f}px)"

                # Продольное смещение головы впепред или назад
                displacement = self.calculate_forward_displacement(landmarks, pose_landmarks, frame.shape)

                if self.reference_displacement is None:
                    self.reference_displacement = displacement

                shift = displacement - self.reference_displacement
                longitudinal_shift = 1 if shift < 20 or shift > 50 else 0
                results['longitudinal_shift'] = f"{longitudinal_shift} {shift:.2f} pixels"

        return frame, results

    def update_gui(self, frame, results):
        self.canvas.delete("all")
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        if results:
            for key, label in self.stats_labels.items():
                label.config(text=str(results.get(key, "N/A")))

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
