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
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def load_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            image_bgr = cv2.imread(file_path)
            if image_bgr is None:
                print("Ошибка при загрузке изображения.")
                return
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            processed_image, results = self.process_frame(image_rgb)

            self.canvas.update()
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
            self.canvas.imgtk = imgtk
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, anchor=tk.CENTER, image=imgtk)

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
        theta = math.acos(np.clip(cos_theta, -1.0, 1.0))
        return math.degrees(theta)

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
            if angle < 10:
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
        output_frame = frame.copy()
        height, width, _ = frame.shape

        face_results = self.face_mesh.process(frame)
        pose_results = self.pose.process(frame)

        # Если обнаружены лицевые ориентиры
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None

            # 1. Наклон головы к плечу (roll) по положению глаз
            left_eye = (face_landmarks[33].x * width, face_landmarks[33].y * height)
            right_eye = (face_landmarks[263].x * width, face_landmarks[263].y * height)
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            tilt_angle = math.degrees(math.atan2(dy, dx))
            if -10 <= tilt_angle <= 10:
                head_tilt_direction = "Нормальное положение"
            elif tilt_angle < -10:
                head_tilt_direction = "Наклон вправо"
            else:
                head_tilt_direction = "Наклон влево"
            results['tilt'] = f"{tilt_angle:.2f}° {head_tilt_direction}"

            # 2. Наклон головы вперед/назад (flexion)
            nose_tip = (face_landmarks[1].x * width, face_landmarks[1].y * height)
            chin = (face_landmarks[152].x * width, face_landmarks[152].y * height)
            face_vector = np.array([chin[0] - nose_tip[0], chin[1] - nose_tip[1]])
            vertical_vector = np.array([0, 1])
            flexion_angle = self.calculate_angle(face_vector, vertical_vector)
            flexion_direction = "Наклон вперед" if face_vector[1] > 0 else "Наклон назад"
            severity, description = self.classify_head_tilt(flexion_angle, flexion_direction)
            results['flexion'] = f"{flexion_angle:.2f}° {flexion_direction} - Severity: {severity} ({description})"

            # 3. Боковое смещение (lateral shift) по новому принципу:
            # Рассчитываем смещение по оси X (от носа до средней точки плеч),
            # нормируя его по ширине плеч.
            if pose_landmarks:
                left_shoulder = np.array([pose_landmarks[11].x * width, pose_landmarks[11].y * height])
                right_shoulder = np.array([pose_landmarks[12].x * width, pose_landmarks[12].y * height])
                mid_shoulder = (left_shoulder + right_shoulder) / 2
                shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                lateral_distance = nose_tip[0] - mid_shoulder[0]
                ratio_lateral = lateral_distance / shoulder_width
                # Порог можно подбирать экспериментально, здесь 0.1 как пример
                if ratio_lateral > 0.1:
                    lateral_dir = "Right"
                elif ratio_lateral < -0.1:
                    lateral_dir = "Left"
                else:
                    lateral_dir = "Neutral"
                results['lateral_shift'] = f"Ratio: {ratio_lateral:.2f} ({lateral_dir})"
            else:
                # Если данные позы недоступны, рассчитываем по абсолютной разнице
                mid_shoulder = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
                lateral_distance = nose_tip[0] - mid_shoulder[0]
                lateral_dir = "Right" if lateral_distance > 0 else "Left"
                results['lateral_shift'] = f"{abs(lateral_distance):.2f}px {lateral_dir}"

            # 4. Продольное смещение: нормированное отношение расстояния от носа до mid_shoulder к ширине плеч
            if pose_landmarks:
                shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
                longitudinal_distance = np.linalg.norm(np.array(nose_tip) - np.array(mid_shoulder))
                ratio_longitudinal = longitudinal_distance / shoulder_distance
                if ratio_longitudinal < 0.95:
                    longitudinal_dir = "Forward"
                elif ratio_longitudinal > 1.05:
                    longitudinal_dir = "Backward"
                else:
                    longitudinal_dir = "Neutral"
                results['longitudinal_shift'] = f"Ratio: {ratio_longitudinal:.2f} ({longitudinal_dir})"
            else:
                results['longitudinal_shift'] = "N/A"

            # 5. Поворот головы (yaw) по расстояниям от носа до ушей
            try:
                left_ear = (face_landmarks[234].x * width, face_landmarks[234].y * height)
                right_ear = (face_landmarks[454].x * width, face_landmarks[454].y * height)
                dist_left = np.linalg.norm(np.array(nose_tip) - np.array(left_ear))
                dist_right = np.linalg.norm(np.array(nose_tip) - np.array(right_ear))
                ratio_ears = (dist_left - dist_right) / ((dist_left + dist_right) / 2)
                yaw_angle = math.degrees(math.atan(ratio_ears))
                if abs(ratio_ears) < 0.1:
                    yaw_dir = "Нейтральное положение"
                elif ratio_ears > 0:
                    yaw_dir = "Поворот налево"
                else:
                    yaw_dir = "Поворот направо"
                results['rotation'] = f"{abs(yaw_angle):.2f}° {yaw_dir}"
            except Exception:
                results['rotation'] = "N/A"

            # Отрисовка реперных точек: уши, подбородок, "кадык" и плечи
            try:
                cv2.circle(output_frame, (int(left_ear[0]), int(left_ear[1])), 4, (0, 255, 0), -1)
                cv2.circle(output_frame, (int(right_ear[0]), int(right_ear[1])), 4, (0, 255, 0), -1)
            except Exception:
                pass
            cv2.circle(output_frame, (int(chin[0]), int(chin[1])), 4, (255, 0, 0), -1)
            kadyk = ((chin[0] + mid_shoulder[0]) / 2, (chin[1] + mid_shoulder[1]) / 2)
            cv2.circle(output_frame, (int(kadyk[0]), int(kadyk[1])), 4, (0, 0, 255), -1)
            if pose_landmarks:
                cv2.circle(output_frame, (int(left_shoulder[0]), int(left_shoulder[1])), 4, (255, 255, 0), -1)
                cv2.circle(output_frame, (int(right_shoulder[0]), int(right_shoulder[1])), 4, (255, 255, 0), -1)
        
        # Если лицевые ориентиры отсутствуют, но есть данные позы
        elif pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            nose = (pose_landmarks[0].x * width, pose_landmarks[0].y * height)
            left_shoulder = np.array([pose_landmarks[11].x * width, pose_landmarks[11].y * height])
            right_shoulder = np.array([pose_landmarks[12].x * width, pose_landmarks[12].y * height])
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            lateral_distance = nose[0] - mid_shoulder[0]
            ratio_lateral = lateral_distance / shoulder_distance
            if ratio_lateral > 0.1:
                lateral_dir = "Right"
            elif ratio_lateral < -0.1:
                lateral_dir = "Left"
            else:
                lateral_dir = "Neutral"
            results['lateral_shift'] = f"Ratio: {ratio_lateral:.2f} ({lateral_dir})"
            longitudinal_distance = np.linalg.norm(np.array(nose) - np.array(mid_shoulder))
            ratio_longitudinal = longitudinal_distance / shoulder_distance
            if ratio_longitudinal < 0.95:
                longitudinal_dir = "Forward"
            elif ratio_longitudinal > 1.05:
                longitudinal_dir = "Backward"
            else:
                longitudinal_dir = "Neutral"
            results['longitudinal_shift'] = f"Ratio: {ratio_longitudinal:.2f} ({longitudinal_dir})"
            face_vector = np.array(nose) - np.array(mid_shoulder)
            vertical_vector = np.array([0, 1])
            flexion_angle = self.calculate_angle(face_vector, vertical_vector)
            flexion_direction = "Наклон вперед" if face_vector[1] > 0 else "Наклон назад"
            severity, description = self.classify_head_tilt(flexion_angle, flexion_direction)
            results['flexion'] = f"{flexion_angle:.2f}° {flexion_direction} - Severity: {severity} ({description})"
            results['rotation'] = "N/A"
            cv2.circle(output_frame, (int(nose[0]), int(nose[1])), 4, (0, 255, 255), -1)
            cv2.circle(output_frame, (int(left_shoulder[0]), int(left_shoulder[1])), 4, (255, 255, 0), -1)
            cv2.circle(output_frame, (int(right_shoulder[0]), int(right_shoulder[1])), 4, (255, 255, 0), -1)
        else:
            results = {
                'tilt': "N/A",
                'flexion': "N/A",
                'lateral_shift': "N/A",
                'longitudinal_shift': "N/A",
                'rotation': "N/A"
            }
        return output_frame, results

    def on_closing(self):
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CervicalDystoniaAnalyzer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
