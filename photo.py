import os
import sys
import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import mediapipe as mp
import math
import numpy as np
from PIL import Image, ImageTk

# Лицензионные настройки
VALID_KEYS = {
    "A1B2C3D4-E5F6-7890-ABCD",
    "C3D4E5F6-A1B2-3456-7890",
    "9876FEDC-BA09-8765-4321",
    "112FEE44-5566-7788-99AA",
    "00FFEEDD-CCBB-AA99-8877",
    "1A2B3C4D-5E6F-7081-92A3",
    "FFEEDDCC-BBAA-9988-7766",
    "A49BCCDD-EEFF-0011-2233",
    "19BC5FF8-90AB-CDEF-1834",
    "DEMO-30DAYS-KEY"
}
DEMO_PERIOD_DAYS = 30
LICENSE_FOLDER = os.path.join(os.getenv("APPDATA"), "CervicalDystoniaAnalyzer")
LICENSE_FILE = os.path.join(LICENSE_FOLDER, "license.bin")
USED_KEYS_FILE = os.path.join(LICENSE_FOLDER, "used_keys.bin")
XOR_KEY = b'yord_tech'


def xor_encrypt(data: bytes, key: bytes) -> bytes:
    """Шифрует/расшифровывает данные с использованием XOR с повторяющимся ключом."""
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


def load_used_keys():
    """
    Загружает список уже использованных ключей из зашифрованного бинарного файла.
    """
    if os.path.exists(USED_KEYS_FILE):
        try:
            with open(USED_KEYS_FILE, "rb") as f:
                encrypted_data = f.read()
            if encrypted_data:
                decrypted_data = xor_encrypt(encrypted_data, XOR_KEY)
                # Файл хранит ключи в виде строк, разделённых переводами строки
                keys = {line.strip() for line in decrypted_data.decode("utf-8").splitlines() if line.strip()}
                return keys
            else:
                return set()
        except Exception:
            return set()
    else:
        return set()


def add_used_key(key):
    """
    Добавляет использованный ключ в зашифрованный бинарный файл.
    """
    used_keys = load_used_keys()
    used_keys.add(key)
    data = "\n".join(used_keys)
    encrypted_data = xor_encrypt(data.encode("utf-8"), XOR_KEY)
    with open(USED_KEYS_FILE, "wb") as f:
        f.write(encrypted_data)


def prompt_for_key():
    """
    Запрашивает у пользователя лицензионный ключ через диалог.
    Возвращает введённый ключ или None, если пользователь отменил ввод.
    """
    temp_root = tk.Tk()
    temp_root.withdraw()
    user_key = simpledialog.askstring("Лицензионный ключ", "Введите лицензионный ключ:")
    temp_root.destroy()
    return user_key


def save_license(license_data: str):
    """
    Сохраняет данные лицензии (ключ, дату активации, дату последнего запуска)
    в зашифрованном виде.
    """
    encrypted_data = xor_encrypt(license_data.encode("utf-8"), XOR_KEY)
    with open(LICENSE_FILE, "wb") as f:
        f.write(encrypted_data)


def load_license():
    """
    Загружает данные лицензии из зашифрованного файла.
    Возвращает список строк или None, если произошла ошибка.
    """
    if os.path.exists(LICENSE_FILE):
        try:
            with open(LICENSE_FILE, "rb") as f:
                encrypted_data = f.read()
            if encrypted_data:
                decrypted_data = xor_encrypt(encrypted_data, XOR_KEY)
                lines = decrypted_data.decode("utf-8").splitlines()
                return lines
            else:
                return None
        except Exception:
            return None
    return None


def activate_license():
    """
    Запрашивает у пользователя лицензионный ключ и активирует лицензию,
    если ключ допустим и ещё не использован. Записывает зашифрованную лицензию,
    содержащую лицензионный ключ, дату активации и дату последнего запуска.
    """
    used_keys = load_used_keys()
    user_key = prompt_for_key()
    if user_key is None:
        messagebox.showerror("Ошибка лицензии", "Лицензионный ключ не введён. Приложение будет закрыто.")
        sys.exit(1)
    if user_key not in VALID_KEYS:
        messagebox.showerror("Ошибка лицензии", "Неверный лицензионный ключ.")
        return activate_license()
    if user_key in used_keys:
        messagebox.showerror("Ошибка лицензии", "Этот лицензионный ключ уже был использован.")
        return activate_license()
    today = datetime.datetime.now().date()
    license_data = f"{user_key}\n{today.isoformat()}\n{today.isoformat()}"
    save_license(license_data)
    add_used_key(user_key)
    return DEMO_PERIOD_DAYS


def check_license():
    """
    Проверяет лицензию: если программа запускается впервые или срок действия лицензии истёк,
    запрашивает лицензионный ключ. Лицензия действительна в течение DEMO_PERIOD_DAYS дней с момента активации.
    Если система изменяет дату назад, используется максимально ранее зафиксированная дата последнего запуска.
    Возвращает количество оставшихся дней демо версии.
    """
    if not os.path.exists(LICENSE_FOLDER):
        os.makedirs(LICENSE_FOLDER)

    today = datetime.datetime.now().date()

    license_lines = load_license()
    if license_lines is None or len(license_lines) < 3:
        days_remaining = activate_license()
        return days_remaining

    try:
        saved_key = license_lines[0].strip()
        activation_date = datetime.datetime.strptime(license_lines[1].strip(), "%Y-%m-%d").date()
        last_run_date = datetime.datetime.strptime(license_lines[2].strip(), "%Y-%m-%d").date()
    except Exception as e:
        messagebox.showerror("Ошибка лицензии", f"Не удалось проверить лицензию: {e}")
        sys.exit(1)

    # Если системная дата откатилась назад, используем last_run_date в качестве текущей
    effective_date = today if today >= last_run_date else last_run_date

    # Если сегодня позже, обновляем last_run_date в файле
    if today > last_run_date:
        try:
            new_license_data = f"{saved_key}\n{activation_date.isoformat()}\n{today.isoformat()}"
            save_license(new_license_data)
            effective_date = today
        except Exception as e:
            messagebox.showerror("Ошибка лицензии", f"Не удалось обновить дату последнего запуска: {e}")
            sys.exit(1)
    
    delta_days = (effective_date - activation_date).days
    days_remaining = DEMO_PERIOD_DAYS - delta_days

    # Если срок действия лицензии истёк, запрашиваем новый ключ
    if days_remaining <= 0:
        messagebox.showinfo("Демо версия истекла", "Срок действия демо версии истёк. Для продолжения работы введите новый лицензионный ключ.")
        days_remaining = activate_license()
    return days_remaining

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

class CervicalDystoniaAnalyzer:
    def __init__(self, root, days_remaining):
        self.root = root
        # Заголовок окна включает информацию о оставшихся днях демо версии
        self.root.title(f"Анализатор кривошеи по фото (Демо версия, осталось {days_remaining} дней)")
        self.setup_ui()
        self.setup_mediapipe()

    def setup_ui(self):
        self.photo_frame = ttk.LabelFrame(self.root, text="Фото")
        self.photo_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.photo_frame, width=640, height=480, bg='gray')
        self.canvas.pack()

        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(padx=10, pady=5, fill=tk.X)
        self.btn_load = ttk.Button(self.control_frame, text="Загрузить фото", command=self.load_photo)
        self.btn_load.pack(side=tk.LEFT, padx=5)

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

    def classify_head_flexion(self, angle, direction):
        if direction == "Наклон вперед":
            if angle < 10:
                return 0, "Отсутствует"
            elif angle < 30:
                return 1, "Лёгкий"
            elif angle < 45:
                return 2, "Умеренный"
            else:
                return 3, "Выраженный"
        elif direction == "Наклон назад":
            if angle < 10:
                return 0, "Отсутствует"
            elif angle < 30:
                return 1, "Лёгкое отклонение"
            elif angle < 45:
                return 2, "Умеренное"
            else:
                return 3, "Выраженное"
        return 0, "N/A"

    def process_frame(self, frame):
        results = {}
        output_frame = frame.copy()
        height, width, _ = frame.shape

        face_results = self.face_mesh.process(frame)
        pose_results = self.pose.process(frame)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None

            left_eye = (face_landmarks[33].x * width, face_landmarks[33].y * height)
            right_eye = (face_landmarks[263].x * width, face_landmarks[263].y * height)
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            tilt_angle = math.degrees(math.atan2(dy, dx))
            abs_tilt = abs(tilt_angle)

            if abs_tilt < 1:
                tilt_description = "Отсутствует"
            elif abs_tilt <= 15:
                tilt_description = "Лёгкий"
            elif abs_tilt <= 35:
                tilt_description = "Умеренный"
            else:
                tilt_description = "Выраженный"
            if tilt_angle < 0:
                tilt_direction = "Наклон вправо"
            elif tilt_angle > 0:
                tilt_direction = "Наклон влево"
            else:
                tilt_direction = ""
            results['tilt'] = f"{abs_tilt:.2f}° {tilt_description} {tilt_direction}"

            nose_tip = (face_landmarks[1].x * width, face_landmarks[1].y * height)
            chin = (face_landmarks[152].x * width, face_landmarks[152].y * height)
            face_vector = np.array([chin[0] - nose_tip[0], chin[1] - nose_tip[1]])
            vertical_vector = np.array([0, 1])
            flexion_angle = self.calculate_angle(face_vector, vertical_vector)
            flexion_direction = "Наклон вперед" if face_vector[1] > 0 else "Наклон назад"
            _, description = self.classify_head_flexion(flexion_angle, flexion_direction)
            results['flexion'] = f"{flexion_angle:.2f}° {description} {flexion_direction}"

            if pose_landmarks:
                left_shoulder = np.array([pose_landmarks[11].x * width, pose_landmarks[11].y * height])
                right_shoulder = np.array([pose_landmarks[12].x * width, pose_landmarks[12].y * height])
                mid_shoulder = (left_shoulder + right_shoulder) / 2
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                lateral_distance = nose_tip[0] - mid_shoulder[0]
                ratio_lateral = lateral_distance / shoulder_width
                results['lateral_shift'] = "0 – Отсутствует" if abs(ratio_lateral) < 0.1 else "1 – Присутствует"
            else:
                mid_shoulder = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
                lateral_distance = nose_tip[0] - mid_shoulder[0]
                results['lateral_shift'] = "0 – Отсутствует" if abs(lateral_distance) < 5 else "1 – Присутствует"

            if pose_landmarks:
                left_shoulder = np.array([pose_landmarks[11].x * width, pose_landmarks[11].y * height])
                right_shoulder = np.array([pose_landmarks[12].x * width, pose_landmarks[12].y * height])
                mid_shoulder = (left_shoulder + right_shoulder) / 2
                shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
                longitudinal_distance = np.linalg.norm(np.array(nose_tip) - np.array(mid_shoulder))
                ratio_longitudinal = longitudinal_distance / shoulder_width
                results['longitudinal_shift'] = "0 – Отсутствует" if 0.95 <= ratio_longitudinal <= 1.05 else "1 – Присутствует"
            else:
                results['longitudinal_shift'] = "N/A"

            try:
                left_ear = (face_landmarks[234].x * width, face_landmarks[234].y * height)
                right_ear = (face_landmarks[454].x * width, face_landmarks[454].y * height)
                dist_left = np.linalg.norm(np.array(nose_tip) - np.array(left_ear))
                dist_right = np.linalg.norm(np.array(nose_tip) - np.array(right_ear))
                ratio_ears = (dist_left - dist_right) / ((dist_left + dist_right) / 2)
                yaw_angle = abs(math.degrees(math.atan(ratio_ears)))
                if yaw_angle < 1:
                    rotation_description = "Отсутствует"
                elif yaw_angle <= 22:
                    rotation_description = "Незначительная"
                elif yaw_angle <= 45:
                    rotation_description = "Лёгкая"
                elif yaw_angle <= 67:
                    rotation_description = "Умеренная"
                else:
                    rotation_description = "Выраженная"
                if ratio_ears > 0:
                    yaw_direction = "Поворот налево"
                elif ratio_ears < 0:
                    yaw_direction = "Поворот направо"
                else:
                    yaw_direction = ""
                results['rotation'] = f"{yaw_angle:.2f}° {rotation_description} {yaw_direction}"
            except Exception:
                results['rotation'] = "N/A"

            try:
                cv2.circle(output_frame, (int(left_ear[0]), int(left_ear[1])), 4, (0, 255, 0), -1)
                cv2.circle(output_frame, (int(right_ear[0]), int(right_ear[1])), 4, (0, 255, 0), -1)
            except Exception:
                pass
            cv2.circle(output_frame, (int(chin[0]), int(chin[1])), 4, (255, 0, 0), -1)
            mid_shoulder = (np.array(left_eye) + np.array(right_eye)) / 2
            kadyk = ((chin[0] + mid_shoulder[0]) / 2, (chin[1] + mid_shoulder[1]) / 2)
            cv2.circle(output_frame, (int(kadyk[0]), int(kadyk[1])), 4, (0, 0, 255), -1)
            if pose_landmarks:
                cv2.circle(output_frame, (int(left_shoulder[0]), int(left_shoulder[1])), 4, (255, 255, 0), -1)
                cv2.circle(output_frame, (int(right_shoulder[0]), int(right_shoulder[1])), 4, (255, 255, 0), -1)
        
        elif pose_results.pose_landmarks:
            pose_landmarks = pose_results.pose_landmarks.landmark
            nose = (pose_landmarks[0].x * width, pose_landmarks[0].y * height)
            left_shoulder = np.array([pose_landmarks[11].x * width, pose_landmarks[11].y * height])
            right_shoulder = np.array([pose_landmarks[12].x * width, pose_landmarks[12].y * height])
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            shoulder_distance = np.linalg.norm(left_shoulder - right_shoulder)
            lateral_distance = nose[0] - mid_shoulder[0]
            ratio_lateral = lateral_distance / shoulder_distance
            results['lateral_shift'] = "0 – Отсутствует" if abs(ratio_lateral) < 0.1 else "1 – Присутствует"

            longitudinal_distance = np.linalg.norm(np.array(nose) - np.array(mid_shoulder))
            ratio_longitudinal = longitudinal_distance / shoulder_distance
            results['longitudinal_shift'] = "0 – Отсутствует" if 0.95 <= ratio_longitudinal <= 1.05 else "1 – Присутствует"
            face_vector = np.array(nose) - np.array(mid_shoulder)
            vertical_vector = np.array([0, 1])
            flexion_angle = self.calculate_angle(face_vector, vertical_vector)
            flexion_direction = "Наклон вперед" if face_vector[1] > 0 else "Наклон назад"
            _, description = self.classify_head_flexion(flexion_angle, flexion_direction)
            results['flexion'] = f"{flexion_angle:.2f}° {description} {flexion_direction}"
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
    # Проверка лицензии и получение оставшихся дней демо версии
    days_remaining = check_license()
    
    root = tk.Tk()
    app = CervicalDystoniaAnalyzer(root, days_remaining)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
