from PyInstaller.utils.hooks import collect_data_files

# Собираем все файлы данных из пакета mediapipe
datas = collect_data_files('mediapipe')