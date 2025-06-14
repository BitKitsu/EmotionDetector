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
    HAPPY = "Szczęście"
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
        
        # Wagi dla różnych emocji (można dostosować)
        self.EMOTION_WEIGHTS = {
            'smile': {'HAPPY': 1.0},                       # bez udziału NEUTRAL
            'surprise': {'SURPRISED': 0.9, 'FEAR': 0.1},
            'brow_furrow': {'ANGRY': 0.6, 'SAD': 0.4},
            'eye_open': {'SURPRISED': 0.8, 'FEAR': 0.2},
            'mouth_open': {'SURPRISED': 0.8, 'HAPPY': 0.2},
            'lip_corner_depression': {'SAD': 0.4, 'ANGRY': 0.3, 'NEUTRAL': 0.3},
            'nose_wrinkle': {'DISGUSTED': 0.9, 'ANGRY': 0.1}
        }
    
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
        """Normalizuje cechy do zakresu [0, 1]."""
        height, width = frame_shape[:2]
        face_size = np.sqrt(width * height)  # Przybliżony rozmiar twarzy
        
        # Normalizacja cech
        normalized = {}
        normalized['smile'] = np.clip(features['smile'] / (0.4 * width), 0, 1)
        normalized['mouth_open'] = np.clip(features['mouth_open'] / (0.3 * height), 0, 1)
        normalized['brow_raise'] = np.clip(features['brow_raise'] / (0.1 * height), 0, 1)
        normalized['brow_furrow'] = np.clip(features['brow_furrow'] / (0.5 * face_size), 0, 1)
        normalized['eye_open'] = np.clip(features['eye_open'] / (0.15 * height), 0, 1)
        normalized['lip_corner_depression'] = np.clip(features['lip_corner_depression'] / (0.3 * height), 0, 1)
        normalized['nose_wrinkle'] = np.clip(features['nose_wrinkle'] / (0.2 * face_size), 0, 1)
        
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
    
    def detect_emotions(self, frame: np.ndarray) -> List[EmotionResult]:
        """Wykrywa emocje na podanym obrazie.
        
        Args:
            frame: Obraz wejściowy w formacie BGR
            
        Returns:
            Lista obiektów EmotionResult z wykrytymi emocjami
        """
        try:
            # Konwersja do RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Wykrywanie twarzy
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return []
            
            results_list = []
            
            for face_landmarks in results.multi_face_landmarks:
                # Obliczenie cech
                distances = self._calculate_distances(face_landmarks.landmark, frame.shape)
                features = self._normalize_features(distances, frame.shape)
                
                # Obliczenie wyników emocji
                emotion_scores = self._calculate_emotion_scores(features)
                
                # Wybór dominującej emocji
                dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                emotion_type = EmotionType[dominant_emotion[0]]
                
                results_list.append(EmotionResult(
                    emotion=emotion_type,
                    confidence=float(dominant_emotion[1]),
                    scores=emotion_scores
                ))
            
            return results_list
            
        except Exception as e:
            logger.error(f"Błąd podczas wykrywania emocji: {e}", exc_info=True)
            return []
    
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
