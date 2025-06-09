import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum

from face_utils import FaceRecognition
from emotion_analyzer import EmotionAnalyzer, EmotionType, EmotionResult
from config import DATA_DIR, CAMERA_SETTINGS, UI_SETTINGS, EMOTION_DETECTION_SETTINGS

logger = logging.getLogger(__name__)


class AuthState(Enum):
    NOT_AUTHENTICATED = "Nieuwierzytelniony"
    AUTHENTICATING = "Uwierzytelnianie..."
    AUTHENTICATED = "Uwierzytelniony"
    REGISTERING = "Rejestracja"
    ERROR = "Błąd"


@dataclass
class UserSession:
    """Klasa przechowująca informacje o sesji użytkownika."""
    user_id: str = ""
    auth_state: AuthState = AuthState.NOT_AUTHENTICATED
    last_seen: float = field(default_factory=time.time)
    emotion_history: List[Dict[str, float]] = field(default_factory=list)
    last_emotions: List[EmotionResult] = field(default_factory=list)
    confidence: float = 0.0
    auth_attempts: int = 0
    max_auth_attempts: int = 3
    session_start: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje obiekt sesji na słownik."""
        return {
            'user_id': self.user_id,
            'auth_state': self.auth_state.value,
            'last_seen': self.last_seen,
            'confidence': self.confidence,
            'auth_attempts': self.auth_attempts,
            'session_start': self.session_start,
            'emotion_history': [
                {e_type.name: score for e_type, score in emotions.items()}
                for emotions in self.emotion_history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserSession':
        """Tworzy obiekt sesji ze słownika."""
        session = cls()
        session.user_id = data.get('user_id', '')
        session.auth_state = AuthState(data.get('auth_state', AuthState.NOT_AUTHENTICATED.value))
        session.last_seen = data.get('last_seen', time.time())
        session.confidence = data.get('confidence', 0.0)
        session.auth_attempts = data.get('auth_attempts', 0)
        session.session_start = data.get('session_start', time.time())
        
        # Konwersja słownika emocji z powrotem na obiekty EmotionResult
        emotion_history = []
        for emotions in data.get('emotion_history', []):
            emotion_scores = {EmotionType[e_type]: score for e_type, score in emotions.items()}
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            emotion_history.append({
                'emotion': dominant_emotion[0],
                'confidence': float(dominant_emotion[1]),
                'scores': emotion_scores
            })
        
        session.emotion_history = emotion_history
        return session


class BiometricSystem:
    def __init__(self):
        """Inicjalizacja systemu biometrycznego."""
        # Inicjalizacja komponentów
        self.face_recognition = FaceRecognition()
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Stan systemu
        self.current_session: Optional[UserSession] = None
        self.registered_users = set()
        self.sessions: Dict[str, UserSession] = {}
        self.last_update_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
        # Ustawienia
        self.min_confidence = 0.5  # Minimalna pewność do rozważenia rozpoznania
        self.max_session_age = 3600  # 1 godzina w sekundach
        self.emotion_history_size = 10  # Liczba przechowywanych wyników emocji
        self.match_threshold = 0.6  # Próg pewności dla rozpoznania (0-1)
        self.required_confirmations = 3  # Liczba potwierdzeń potrzebnych do uwierzytelnienia
        self.confirmation_count = 0  # Licznik potwierdzeń
        self.last_authentication_time = 0  # Czas ostatniego udanego uwierzytelnienia
        
        # Wczytanie zapisanych sesji
        self._load_sessions()
    
    def _load_sessions(self) -> None:
        """Wczytuje zapisane sesje użytkowników."""
        sessions_file = DATA_DIR / 'user_sessions.json'
        if sessions_file.exists():
            try:
                with open(sessions_file, 'r') as f:
                    sessions_data = json.load(f)
                    self.sessions = {
                        user_id: UserSession.from_dict(session_data)
                        for user_id, session_data in sessions_data.items()
                    }
                logger.info(f"Wczytano {len(self.sessions)} sesji użytkowników")
            except Exception as e:
                logger.error(f"Błąd podczas wczytywania sesji: {e}")
    
    def _save_sessions(self) -> None:
        """Zapisuje sesje użytkowników do pliku."""
        sessions_file = DATA_DIR / 'user_sessions.json'
        try:
            sessions_data = {
                user_id: session.to_dict()
                for user_id, session in self.sessions.items()
            }
            with open(sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania sesji: {e}")
    
    def _update_fps(self) -> None:
        """Aktualizuje liczbę klatek na sekundę."""
        self.frame_count += 1
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        if time_elapsed >= 1.0:  # Aktualizuj FPS co sekundę
            self.fps = self.frame_count / time_elapsed
            self.frame_count = 0
            self.last_update_time = current_time
    
    def register_user(self, user_id: str, image: np.ndarray) -> bool:
        """Rejestruje nowego użytkownika.
        
        Args:
            user_id: Unikalny identyfikator użytkownika
            image: Obraz twarzy do rejestracji
            
        Returns:
            bool: True jeśli rejestracja się powiodła, False w przeciwnym wypadku
        """
        if not user_id or not user_id.strip():
            logger.error("Nieprawidłowy identyfikator użytkownika")
            return False
            
        if user_id in self.sessions:
            logger.warning(f"Użytkownik {user_id} jest już zarejestrowany")
            return False
        
        # Rejestracja twarzy
        success = self.face_recognition.register_face(image, user_id)
        
        if success:
            # Utworzenie nowej sesji
            self.sessions[user_id] = UserSession(
                user_id=user_id,
                auth_state=AuthState.AUTHENTICATED,
                session_start=time.time()
            )
            self.registered_users.add(user_id)
            self._save_sessions()
            logger.info(f"Zarejestrowano nowego użytkownika: {user_id}")
            return True
        
        return False
    
    def authenticate_user(self, frame: np.ndarray) -> Optional[UserSession]:
        """Uwierzytelnia użytkownika na podstawie twarzy i emocji.
        
        Args:
            frame: Obraz z kamerki
            
        Returns:
            Obiekt UserSession jeśli uwierzytelnienie się powiodło, None w przeciwnym wypadku
        """
        # Aktualizacja FPS
        self._update_fps()
        
        # Wykrywanie i rozpoznawanie twarzy
        face_results = self.face_recognition.recognize_face(frame)
        
        if not face_results:
            self.current_session = None
            return None
        
        # Pobranie najlepszego dopasowania
        best_match = max(face_results, key=lambda x: x[1])
        user_id, confidence, (top, right, bottom, left) = best_match
        
        # Jeśli pewność jest zbyt niska, traktuj jako nieznanego użytkownika
        if confidence < self.min_confidence:
            logger.debug(f"Zbyt niska pewność rozpoznania: {confidence:.2f} < {self.min_confidence}")
            user_id = "Nieznany"
            confidence = 0.0  # Upewnij się, że confidence jest ustawione na 0 dla nieznanego użytkownika
        
        # Analiza emocji
        emotion_results = self.emotion_analyzer.detect_emotions(frame)
        
        # Pobranie lub utworzenie sesji użytkownika
        if user_id in self.sessions:
            session = self.sessions[user_id]
        else:
            session = UserSession(
                user_id=user_id,
                auth_state=AuthState.NOT_AUTHENTICATED,
                last_seen=time.time()
            )
            self.sessions[user_id] = session

        # Odśwież znacznik czasu
        current_time = time.time()
        session.last_seen = current_time
        
        # Sprawdź, czy użytkownik jest już uwierzytelniony i czy sesja jest wciąż ważna
        if session.auth_state == AuthState.AUTHENTICATED:
            # Sprawdź, czy sesja wygasła
            if current_time - session.last_seen > self.max_session_age:
                session.auth_state = AuthState.UNAUTHENTICATED
                logger.info(f"Sesja wygasła dla użytkownika: {user_id}")
            else:
                # Aktualizuj czas ostatnio widzianego
                session.last_seen = current_time
                return session
        
        # Logika uwierzytelniania dla nieuwierzytelnionych użytkowników
        if confidence >= self.min_confidence and session.auth_state != AuthState.AUTHENTICATED:
            logger.debug(f"Potencjalne dopasowanie: {user_id} (pewność: {confidence:.2f}, próg: {self.match_threshold})")
            
            if confidence >= self.match_threshold:
                self.confirmation_count += 1
                logger.debug(f"Dobre dopasowanie: {self.confirmation_count}/{self.required_confirmations}")
                
                if self.confirmation_count >= self.required_confirmations:
                    session.auth_state = AuthState.AUTHENTICATED
                    session.confidence = confidence
                    session.last_seen = current_time
                    self.last_authentication_time = current_time
                    self.confirmation_count = 0
                    logger.info(f"Użytkownik {user_id} pomyślnie uwierzytelniony (pewność: {confidence:.2f})")
            else:
                # Resetuj licznik jeśli pewność spadnie poniżej progu
                if self.confirmation_count > 0:
                    logger.debug("Resetowanie licznika potwierdzeń - zbyt niska pewność")
                    self.confirmation_count = 0
        
        # Resetuj licznik jeśli minęło dużo czasu od ostatniego dobrego dopasowania
        if current_time - self.last_authentication_time > 5.0:  # 5 sekund
            if self.confirmation_count > 0:
                logger.debug("Resetowanie licznika potwierdzeń - przekroczono czas")
                self.confirmation_count = 0
            
            # Jeśli użytkownik jest uwierzytelniony, aktualizuj historię emocji
            if session.auth_state == AuthState.AUTHENTICATED:
                if emotion_results:
                    emotion_data = emotion_results[0].scores
                    session.emotion_history.append(emotion_data)
                    session.emotion_history = session.emotion_history[-self.emotion_history_size:]
                    session.last_emotions = emotion_results
        
        self.current_session = session
        return session
    
    def get_emotion_summary(self, user_id: str) -> Dict[str, float]:
        """Zwraca podsumowanie emocji dla danego użytkownika."""
        if user_id not in self.sessions:
            return {}
        
        session = self.sessions[user_id]
        if not session.emotion_history:
            return {}
        
        # Oblicz średnie wartości emocji
        emotion_sums = {}
        for emotions in session.emotion_history:
            for emotion, value in emotions.items():
                if emotion not in emotion_sums:
                    emotion_sums[emotion] = 0.0
                emotion_sums[emotion] += value
        
        # Normalizacja
        total = sum(emotion_sums.values())
        if total > 0:
            emotion_avg = {k.name: v/total for k, v in emotion_sums.items()}
        else:
            emotion_avg = {}
        
        return emotion_avg
    
    def draw_ui(self, frame: np.ndarray, session: Optional[UserSession] = None) -> np.ndarray:
        """Rysuje interfejs użytkownika na ramce wideo."""
        img = frame.copy()
        height, width = img.shape[:2]
        
        # Tło dla paska statusu
        cv2.rectangle(img, (0, 0), (width, 40), (50, 50, 50), -1)
        
        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(img, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        if session:
            # Status uwierzytelnienia
            status_text = f"Status: {session.auth_state.value}"
            status_color = (0, 255, 0) if session.auth_state == AuthState.AUTHENTICATED else (0, 0, 255)
            
            cv2.putText(img, status_text, (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, status_color, 2)
            
            # Informacje o użytkowniku
            user_text = f"Użytkownik: {session.user_id}"
            cv2.putText(img, user_text, (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # Jeśli wykryto emocje, wyświetl dominującą
            if session.last_emotions:
                emotion = session.last_emotions[0].emotion
                confidence = session.last_emotions[0].confidence
                emotion_text = f"{emotion.value}: {confidence:.1%}"
                
                # Kolor w zależności od emocji
                emotion_colors = self.emotion_analyzer.get_emotion_colors()
                color = emotion_colors.get(emotion, (255, 255, 255))
                
                cv2.putText(img, emotion_text, (width - 250, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return img
    
    def cleanup(self) -> None:
        """Zwalnia zasoby systemu."""
        self._save_sessions()
        cv2.destroyAllWindows()
