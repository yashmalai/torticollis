import tkinter as tk
from tkinter import ttk, filedialog
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
        self.root.title("Анализатор кривошеи по фото")

        self.setup_ui()
        self.setup_mediapipe()

    def setup_ui(self):
        # Фрейм для отображения фото
        self.photo_frame = ttk.LabelFrame(self.root, text="Фото")
        self.photo_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.photo_frame, width=640, height=480, bg='gray')
        self.canvas.pack()

        # Фрейм для кнопок управления
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(padx=10, pady=5, fill=tk.X)

        self.btn_load = ttk.Button(self.control_frame, text="Загрузить фото", command=self.load_photo)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        # Фрейм для отображения диагностических параметров
        self.stats_frame = ttk.LabelFrame(self.root, text="Диагностические параметры")
        self.stats_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.stats_labels = {}
        parameters = [
            ("Поворот головы", "rotation"),
            ("Наклон головы к плечу", "tilt"),
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
        # Используем статический режим для обработки одиночного фото
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5)

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5)

    def load_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            image_bgr = cv2.imread(file_path)
            if image_bgr is None:
                print("Ошибка при загрузке изображения.")
                return

            # Конвертируем изображение из BGR в RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            processed_image, results = self.process_frame(image_rgb)

            # Масштабирование изображения для правильного отображения на canvas
            self.canvas.update()  # Обновляем размеры canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            pil_image = Image.fromarray(processed_image)

            image_width, image_height = pil_image.size
            scale = min(canvas_width / image_width, canvas_height / image_height)
            new_width = int(image_width * scale)
            new_height = int(image_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(pil_image)
            self.canvas.delete("all")
            self.canvas.imgtk = imgtk  # Сохраняем ссылку, чтобы изображение не исчезло
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=imgtk)

            # Обновляем диагностические параметры
            if results:
                for key, label in self.stats_labels.items():
                    label.config(text=str(results.get(key, "N/A")))
            else:
                for label in self.stats_labels.values():
                    label.config(text="N/A")

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
        return 0, "N/A"

    def process_frame(self, frame):
        results = {}
        height, width, _ = frame.shape
        face_results = self.face_mesh.process(frame)
        pose_results = self.pose.process(frame)

        if face_results.multi_face_landmarks and pose_results.pose_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = face_landmarks.landmark  # Лицевые ориентиры
                pose_landmarks = pose_results.pose_landmarks.landmark  # Ориентиры тела

                # 1. Наклон головы к плечу (определяется по положению глаз)
                left_eye = (landmarks[33].x, landmarks[33].y)
                right_eye = (landmarks[263].x, landmarks[263].y)
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                tilt = math.degrees(math.atan2(dy, dx))
                if -10 <= tilt <= 10:
                    head_tilt_direction = "Нормальное положение"
                elif tilt < -10:
                    head_tilt_direction = "Наклон вправо"
                else:
                    head_tilt_direction = "Наклон влево"
                results['tilt'] = f"{tilt:.2f}° {head_tilt_direction}"

                # 2. Наклон головы вперед/назад (определяется по соотношению лба, носа и подбородка)
                nose_tip = (int(landmarks[1].x * width), int(landmarks[1].y * height))
                forehead = (int(landmarks[10].x * width), int(landmarks[10].y * height))
                chin = (int(landmarks[152].x * width), int(landmarks[152].y * height))
                face_vector = np.array([chin[0] - forehead[0], chin[1] - forehead[1]])
                vertical_vector = np.array([0, 1])
                angle = self.calculate_angle(face_vector, vertical_vector)
                flexion_direction = "Наклон вперед" if face_vector[0] > 0 else "Наклон назад"
                severity, description = self.classify_head_tilt(angle, flexion_direction)
                results['flexion'] = f"{angle:.2f}° {flexion_direction} - Severity: {severity} ({description})"

                # 3. Боковое смещение головы (рассчитывается по разнице между положением носа и средней точкой плеч)
                nose = np.array([landmarks[1].x * width, landmarks[1].y * height])
                left_shoulder = np.array([pose_landmarks[11].x * width, pose_landmarks[11].y * height])
                right_shoulder = np.array([pose_landmarks[12].x * width, pose_landmarks[12].y * height])
                mid_shoulder = (left_shoulder + right_shoulder) / 2
                lateral_shift = nose[0] - mid_shoulder[0]
                if lateral_shift > 0:
                    lateral_dir = "Right"
                else:
                    lateral_dir = "Left"
                results['lateral_shift'] = f"{abs(lateral_shift):.2f}px {lateral_dir}"

                # 4. Продольное смещение головы (рассчитывается по разнице по вертикали между носом и плечами)
                longitudinal_shift = nose[1] - mid_shoulder[1]
                if longitudinal_shift > 0:
                    longitudinal_dir = "Forward"
                else:
                    longitudinal_dir = "Backward"
                results['longitudinal_shift'] = f"{abs(longitudinal_shift):.2f}px {longitudinal_dir}"

                # 5. Поворот головы (yaw) влево или направо (определяется по расстояниям от носа до ушей)
                try:
                    left_ear = np.array([landmarks[234].x * width, landmarks[234].y * height])
                    right_ear = np.array([landmarks[454].x * width, landmarks[454].y * height])
                    dist_left = np.linalg.norm(nose - left_ear)
                    dist_right = np.linalg.norm(nose - right_ear)
                    # Вычисляем относительную разницу расстояний
                    ratio = (dist_left - dist_right) / ((dist_left + dist_right) / 2)
                    yaw_angle = math.degrees(math.atan(ratio))
                    if abs(ratio) < 0.1:
                        yaw_dir = "Нейтральное положение"
                    elif ratio > 0:
                        yaw_dir = "Поворот налево"
                    else:
                        yaw_dir = "Поворот направо"
                    results['rotation'] = f"{abs(yaw_angle):.2f}° {yaw_dir}"
                except Exception as e:
                    results['rotation'] = "N/A"
        else:
            results = {
                'tilt': "N/A",
                'flexion': "N/A",
                'lateral_shift': "N/A",
                'longitudinal_shift': "N/A",
                'rotation': "N/A"
            }
        return frame, results

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CervicalDystoniaAnalyzer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
