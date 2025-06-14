import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionType(Enum):
    NEUTRAL = "Neutralna"
    HAPPY = "Radość"
    SAD = "Smutek"
    SURPRISED = "Zaskoczenie"
    ANGRY = "Złość"
    DISGUSTED = "Obrzydzenie"
    FEAR = "Strach"

@dataclass
class EmotionResult:
    emotion: EmotionType
    confidence: float
    scores: Dict[str, float]

class EmotionAnalyzer:
    def __init__(self):
        """Inicjalizacja analizatora emocji."""
        # Inicjalizacja MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Punkty charakterystyczne twarzy
        self.LIPS_LEFT = 61
        self.LIPS_RIGHT = 291
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.BROW_LEFT = 70
        self.BROW_RIGHT = 300
        self.EYE_LEFT = 33
        self.EYE_RIGHT = 263
        self.NOSE_TIP = 4
        # --- Kalibracja neutralnej twarzy ---
        self.baseline_raw: Optional[Dict[str, float]] = None
        self._calibrating: bool = False
        self._calib_samples: List[Dict[str, float]] = []
        self._required_samples: int = 25
        self._calibrating_for_user: Optional[str] = None
        
        # Wagi dla różnych emocji (można dostosować)
        self.EMOTION_WEIGHTS = {
            'smile': {'HAPPY': 0.7, 'NEUTRAL': 0.3},
            'surprise': {'SURPRISED': 0.9, 'FEAR': 0.1},
            'brow_furrow': {'ANGRY': 0.6, 'SAD': 0.4},
            'eye_open': {'SURPRISED': 0.7, 'FEAR': 0.3},
            'mouth_open': {'SURPRISED': 0.8, 'HAPPY': 0.2},
            'lip_corner_depression': {'SAD': 0.35, 'ANGRY': 0.2, 'NEUTRAL': 0.45},
            'nose_wrinkle': {'DISGUSTED': 0.9, 'ANGRY': 0.1}
        }

    def set_baseline(self, baseline: Dict[str, float]):
        """Ustawia istniejącą linię bazową dla emocji."""
        self.baseline_raw = baseline
        logger.info("Załadowano istniejącą linię bazową dla emocji.")

    def is_calibrating(self) -> bool:
        """Zwraca, czy trwa proces kalibracji."""
        return self._calibrating
    
    def start_calibration(self, user_id: str, samples: int = 25):
        """Rozpoczyna kalibrację neutralnej twarzy dla danego użytkownika."""
        if not user_id:
            logger.error("Nie można rozpocząć kalibracji bez identyfikatora użytkownika.")
            return

        self.baseline_raw = None  # Resetuj bazę przed nową kalibracją
        self._calibrating = True
        self._calib_samples = []
        self._required_samples = samples
        self._calibrating_for_user = user_id
        logger.info(f"Rozpoczęto kalibrację neutralnej twarzy dla '{user_id}' (zbieranie {samples} próbek)...")

    def abort_calibration(self):
        """Przerywa proces kalibracji."""
        if not self._calibrating:
            return
        logger.warning(f"Przerwano kalibrację dla użytkownika '{self._calibrating_for_user}' z powodu wykrycia innej osoby lub błędu.")
        self._calibrating = False
        self._calib_samples = []
        self._calibrating_for_user = None

    def _get_landmark_point(self, landmarks, idx: int, frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Pobiera współrzędne punktu charakterystycznego."""
        height, width = frame_shape[:2]
        return (landmarks[idx].x * width, landmarks[idx].y * height)
    
    def _calculate_distances(self, landmarks, frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Oblicza odległości między punktami charakterystycznymi twarzy."""
        # Pobranie punktów
        p = lambda idx: self._get_landmark_point(landmarks, idx, frame_shape)
        
        # Obliczenie odległości
        distances = {}
        
        # Uśmiech - szerokość ust
        distances['smile'] = np.linalg.norm(np.array(p(self.LIPS_RIGHT)) - np.array(p(self.LIPS_LEFT)))
        
        # Otwarcie ust
        distances['mouth_open'] = np.linalg.norm(np.array(p(self.MOUTH_BOTTOM)) - np.array(p(self.MOUTH_TOP)))
        
        # Uniesienie brwi
        brow_left = p(self.BROW_LEFT)
        brow_right = p(self.BROW_RIGHT)
        eye_left = p(self.EYE_LEFT)
        eye_right = p(self.EYE_RIGHT)
        
        # Średnia odległość między brwiami a oczami
        distances['brow_raise'] = ((brow_left[1] - eye_left[1]) + (brow_right[1] - eye_right[1])) / 2
        
        # Zmarszczenie brwi
        nose_tip = p(self.NOSE_TIP)
        distances['brow_furrow'] = (np.linalg.norm(np.array(brow_left) - np.array(nose_tip)) + 
                                   np.linalg.norm(np.array(brow_right) - np.array(nose_tip))) / 2
        
        # Otwarcie oczu (uproszczone)
        distances['eye_open'] = (eye_left[1] - brow_left[1] + eye_right[1] - brow_right[1]) / 2
        
        # Opadanie kącików ust
        lip_left = p(self.LIPS_LEFT)
        lip_right = p(self.LIPS_RIGHT)
        distances['lip_corner_depression'] = (lip_left[1] + lip_right[1]) / 2 - (brow_left[1] + brow_right[1]) / 2
        
        # Marszczenie nosa (uproszczone)
        distances['nose_wrinkle'] = np.linalg.norm(np.array(p(19)) - np.array(nose_tip))
        
        return distances
    
    def _normalize_features(self, features: Dict[str, float], frame_shape: Tuple[int, int]) -> Dict[str, float]:
        """Normalizuje cechy (odchyłka od baseline) do zakresu [0, 1]."""
        height, width = frame_shape[:2]
        face_size = np.sqrt(width * height)

        def _norm(val, denom):
            return np.clip(val / denom, 0, 1)

        # Odchyłki względem baseline, jeżeli dostępna
        if self.baseline_raw:
            diff = {k: abs(features[k] - self.baseline_raw.get(k, features[k])) for k in features}
        else:
            diff = features

        normalized = {}
        normalized['smile'] = _norm(diff['smile'], 0.4 * width)
        normalized['mouth_open'] = _norm(diff['mouth_open'], 0.3 * height)
        normalized['brow_raise'] = _norm(diff['brow_raise'], 0.1 * height)
        normalized['brow_furrow'] = _norm(diff['brow_furrow'], 0.5 * face_size)
        normalized['eye_open'] = _norm(diff['eye_open'], 0.15 * height)
        normalized['lip_corner_depression'] = _norm(diff['lip_corner_depression'], 0.3 * height)
        normalized['nose_wrinkle'] = _norm(diff['nose_wrinkle'], 0.2 * face_size)

        return normalized
    
    def _calculate_emotion_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Oblicza wyniki emocji na podstawie znormalizowanych cech."""
        # Inicjalizacja wyników
        emotion_scores = {emotion.name: 0.0 for emotion in EmotionType}
        
        # Obliczenie wyników dla każdej emocji
        for feature_name, feature_value in features.items():
            if feature_name in self.EMOTION_WEIGHTS:
                for emotion, weight in self.EMOTION_WEIGHTS[feature_name].items():
                    emotion_scores[emotion] += feature_value * weight
        
        # Normalizacja wyników do sumy 1.0
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        
        return emotion_scores
    
    def detect_emotions(self, frame: np.ndarray, current_user_id: Optional[str] = None) -> Tuple[List[EmotionResult], Optional[Dict[str, float]]]:
        """
        Wykrywa emocje na podanym obrazie.
        
        Args:
            frame: Obraz wejściowy w formacie BGR
            current_user_id: ID aktualnie wykrytego użytkownika, do weryfikacji podczas kalibracji.
            
        Returns:
            Tuple[List[EmotionResult], Optional[Dict[str, float]]]: Wyniki emocji i nowa linia bazowa (jeśli utworzono).
        """
        # Nie wykrywaj emocji dla nierozpoznanych użytkowników
        if current_user_id == "Nieznany":
            return [], None

        try:
            # Sprawdzenie bezpieczeństwa kalibracji
            if self.is_calibrating():
                if current_user_id != self._calibrating_for_user:
                    self.abort_calibration()
                    return [], None  # Zwróć puste wyniki, kalibracja przerwana

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            new_baseline = None
            if not results.multi_face_landmarks:
                self._maybe_collect_baseline(None)
                return [], None

            landmarks = results.multi_face_landmarks[0].landmark
            raw_features = self._calculate_distances(landmarks, frame.shape)

            new_baseline = self._maybe_collect_baseline(raw_features)

            normalized_features = self._normalize_features(raw_features, frame.shape)
            emotion_scores = self._calculate_emotion_scores(normalized_features)

            if not emotion_scores:
                return [], new_baseline

            dominant_emotion_type = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion_type]

            emotion_results = [EmotionResult(
                emotion=EmotionType[dominant_emotion_type],
                confidence=confidence,
                scores=emotion_scores
            )]

            return emotion_results, new_baseline
        except Exception as e:
            logger.error(f"Błąd podczas wykrywania emocji: {e}", exc_info=True)
            return [], None

    def _maybe_collect_baseline(self, raw_features: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Jeśli trwa kalibracja, zbiera próbkę. Zwraca nową linię bazową po zakończeniu."""
        if not self._calibrating:
            return None

        if raw_features is None:  # Nie wykryto twarzy, nie zbieraj próbki
            return None

        self._calib_samples.append(raw_features)
        logger.info("Zebrano próbkę kalibracyjną %d/%d", len(self._calib_samples), self._required_samples)

        if len(self._calib_samples) >= self._required_samples:
            # Oblicz średnią linię bazową
            avg_baseline = {}
            if not self._calib_samples:
                return None

            for key in self._calib_samples[0]:
                avg_baseline[key] = sum(s[key] for s in self._calib_samples) / len(self._calib_samples)

            self.baseline_raw = avg_baseline
            self._calibrating = False
            logger.info("Zakończono kalibrację. Ustawiono nową bazę: %s", self.baseline_raw)
            return self.baseline_raw  # Zwróć nową linię bazową

        return None

    def draw_landmarks(self, frame: np.ndarray, landmarks, color=(0, 255, 0), thickness=1) -> np.ndarray:
        """Rysuje punkty charakterystyczne na obrazie."""
        if landmarks is None:
            return frame
            
        img = frame.copy()
        height, width = img.shape[:2]
        
        # Rysowanie punktów
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(img, (x, y), 2, color, thickness)
        
        return img

    def get_emotion_colors(self) -> Dict[EmotionType, Tuple[int, int, int]]:
        """Zwraca kolory przypisane do poszczególnych emocji."""
        return {
            EmotionType.NEUTRAL: (200, 200, 200),    # Szary
            EmotionType.HAPPY: (0, 255, 255),         # Żółty
            EmotionType.SAD: (255, 0, 0),             # Niebieski
            EmotionType.SURPRISED: (255, 165, 0),     # Pomarańczowy
            EmotionType.ANGRY: (0, 0, 255),           # Czerwony
            EmotionType.DISGUSTED: (0, 128, 0),       # Zielony
            EmotionType.FEAR: (128, 0, 128)           # Fioletowy
        }
