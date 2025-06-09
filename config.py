import os
from pathlib import Path

# Ścieżki
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Tworzenie katalogów jeśli nie istnieją
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Ustawienia kamery
CAMERA_SETTINGS = {
    'width': 640,
    'height': 480,
    'fps': 30,
}

# Ustawienia wykrywania twarzy
FACE_DETECTION_SETTINGS = {
    'model_path': str(MODELS_DIR / 'face_detection_short_range.sparse.tflite'),
    'min_detection_confidence': 0.5,
}

# Ustawienia rozpoznawania twarzy
FACE_RECOGNITION_SETTINGS = {
    'tolerance': 0.6,  # Tolerancja dla dopasowania twarzy (im mniejsza, tym bardziej restrykcyjne)
    'model': 'hog',  # 'hog' (szybszy) lub 'cnn' (dokładniejszy, ale wymaga CUDA)
}

# Ustawienia wykrywania emocji
EMOTION_DETECTION_SETTINGS = {
    'update_interval': 0.1,  # Częstotliwość aktualizacji wykresu emocji (w sekundach)
    'emotion_labels': ['Uśmiech', 'Zdziwienie', 'Złość', 'Smutek', 'Neutralna'],
}

# Ustawienia interfejsu
UI_SETTINGS = {
    'window_size': (1280, 720),
    'theme': {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'success': '#2ecc71',
        'danger': '#e74c3c',
        'text': '#2c3e50',
        'background': '#ecf0f1',
    },
}

# Ścieżki do modeli
MODEL_URLS = {
    'face_detection': 'https://storage.googleapis.com/mediapipe-models/face_detection/blaze_face_short_range/float16/1/blaze_face_short_range.tflite',
    'face_landmark': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
}

# Ustawienia logowania
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(BASE_DIR / 'app.log'),
            'formatter': 'standard',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
    },
}
