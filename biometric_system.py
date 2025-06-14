import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from enum import Enum

from face_utils import FaceRecognition, DistanceMetric
from emotion_analyzer import EmotionAnalyzer, EmotionType, EmotionResult
from config import DATA_DIR, CAMERA_SETTINGS, UI_SETTINGS, EMOTION_DETECTION_SETTINGS

logger = logging.getLogger(__name__)

# Używamy standardowego cv2.putText z podstawieniem polskich znaków
# zamiast cv2.freetype, które może nie być dostępne we wszystkich instalacjach
logger.info("Using standard OpenCV text rendering with Polish character substitution.")

def _draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], font_scale: float = 0.7,
               color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> None:
    """Rysuje tekst na obrazie z obsługą polskich znaków.
    
    Używa standardowego cv2.putText z podstawieniem znaków specjalnych.
    """
    # Mapa polskich znaków do ich odpowiedników ASCII
    polish_chars = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    }
    
    # Zamień polskie znaki na ich odpowiedniki ASCII
    clean_text = ''.join(polish_chars.get(c, c) for c in text)
    
    # Użyj standardowego putText z czcionką, która obsługuje szeroki zakres znaków
    cv2.putText(img, clean_text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


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
    face_results: List[Tuple[str, float, Tuple[int, int, int, int]]] = field(default_factory=list)
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
        self.start_time = time.time()
        self.fps = 0.0
        self.last_authentication_time = 0.0
        self.last_confirmation_increment_time = 0.0
        self.confirmation_count = 0
        self.required_confirmations = 3
        
        # Ustawienia
        self.max_session_age = 3600  # 1 godzina w sekundach
        self.emotion_history_size = 10  # Liczba przechowywanych wyników emocji
        
        # Inicjalizacja tolerancji w face_recognition na podstawie domyślnych progów pewności
        self._update_face_recognition_tolerance()
        
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
    
    def authenticate_user_with_results(self, frame: np.ndarray, face_results: List[Tuple[str, float, Tuple]]) -> UserSession:
        """Uwierzytelnia użytkownika na podstawie PRZETWORZONYCH wyników rozpoznawania twarzy i emocji.
        
        Args:
            frame: Obraz z kamerki (używany do analizy emocji, wyniki rozpoznawania twarzy są już podane).
            face_results: Lista wyników rozpoznawania twarzy.
            
        Returns:
            Obiekt UserSession.
        """
        if not face_results:
            # Jeśli nie ma twarzy, wyczyść bieżącą sesję, jeśli istnieje
            if self.current_session and self.current_session.user_id != "Nieznany":
                logger.info(f"Użytkownik {self.current_session.user_id} nie jest już widoczny. Zamykanie sesji.")
            self.current_session = None
            # Zwróć nową, pustą sesję, aby uniknąć zwracania None, co powoduje błędy
            return UserSession()
        
        # Pobranie najlepszego dopasowania
        best_match = max(face_results, key=lambda x: x[1])
        user_id, confidence, (top, right, bottom, left) = best_match
        

        
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
        # Zawsze aktualizuj pewność sesji
        session.confidence = confidence
        # Aktualizacja emocji – zapisuj zawsze gdy dostępne
        if emotion_results:
            session.last_emotions = emotion_results
            emotion_data = emotion_results[0].scores
            session.emotion_history.append(emotion_data)
            # Ogranicz historię do ostatnich N wpisów
            session.emotion_history = session.emotion_history[-self.emotion_history_size:]

        # Sprawdź, czy użytkownik jest już uwierzytelniony i czy sesja jest wciąż ważna
        if session.auth_state == AuthState.AUTHENTICATED:
            # Sprawdź, czy sesja wygasła
            if current_time - session.last_seen > self.max_session_age:
                session.auth_state = AuthState.UNAUTHENTICATED
                logger.info(f"Sesja wygasła dla użytkownika: {user_id}")
            # Jeśli sesja jest aktywna, pewność została już zaktualizowana, więc możemy kontynuować
        
        # Logika uwierzytelniania dla nieuwierzytelnionych użytkowników
        if session.auth_state != AuthState.AUTHENTICATED:
            active_metric = self.face_recognition.metric
            confidence_threshold = self.get_match_threshold(active_metric)
            if confidence >= confidence_threshold:
                self.confirmation_count += 1
                self.last_confirmation_increment_time = current_time
                logger.debug(f"Dobre dopasowanie: {self.confirmation_count}/{self.required_confirmations}")
                
                if self.confirmation_count >= self.required_confirmations:
                    session.auth_state = AuthState.AUTHENTICATED
                    self.last_authentication_time = current_time
                    self.confirmation_count = 0
                    self.last_confirmation_increment_time = 0.0
                    logger.info(f"Użytkownik {user_id} pomyślnie uwierzytelniony (pewność: {confidence:.2f})")
            else:
                # Resetuj licznik jeśli pewność spadnie poniżej progu
                if self.confirmation_count > 0:
                    logger.debug("Resetowanie licznika potwierdzeń - zbyt niska pewność")
                    self.confirmation_count = 0
                    self.last_confirmation_increment_time = 0.0
        
        # Resetuj licznik potwierdzeń, jeśli sekwencja utknęła (brak postępu)
        if 0 < self.confirmation_count < self.required_confirmations:
            if current_time - self.last_confirmation_increment_time > 3.0:  # Np. 3 sekundy na kolejny dobry match
                logger.debug(f"Resetowanie licznika potwierdzeń ({self.confirmation_count}) - przekroczono czas (3s) od ostatniego przyrostu.")
                self.confirmation_count = 0
                self.last_confirmation_increment_time = 0.0
            
            # Jeśli użytkownik jest uwierzytelniony, aktualizuj historię emocji

        
        self.current_session = session
        return session
    
    def _update_fps(self):
        """Aktualizuje licznik FPS."""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def authenticate_user(self, frame: np.ndarray) -> UserSession:
        """Uwierzytelnia użytkownika: wykrywa twarze, a następnie używa authenticate_user_with_results."""
        self._update_fps()
        # Optymalizacja: skalowanie klatki przed detekcją
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face_results_small = self.face_recognition.recognize_face(small_frame)

        # Przeskaluj wyniki z powrotem do oryginalnego rozmiaru klatki
        face_results = []
        if face_results_small:
            for name, conf, (top_s, right_s, bottom_s, left_s) in face_results_small:
                top = int(top_s * 2)
                right = int(right_s * 2)
                bottom = int(bottom_s * 2)
                left = int(left_s * 2)

                face_results.append((name, conf, (top, right, bottom, left)))

        session = self.authenticate_user_with_results(frame, face_results)
        session.face_results = face_results
        return session

    def _update_face_recognition_tolerance(self):
        """Aktualizuje tolerancję w module face_recognition na podstawie aktywnej metryki."""
        active_metric = self.face_recognition.metric
        new_tolerance = self.get_match_threshold(active_metric)
        self.face_recognition.set_tolerance(active_metric, new_tolerance)

    def set_match_threshold(self, metric: DistanceMetric, value: float):
        """Ustawia próg tolerancji dla danej metryki w systemie rozpoznawania twarzy."""
        self.face_recognition.set_tolerance(metric, value)

    def get_match_threshold(self, metric: DistanceMetric) -> float:
        """Pobiera próg tolerancji dla danej metryki z systemu rozpoznawania twarzy."""
        return self.face_recognition.get_tolerance(metric)

    def toggle_metric(self) -> 'DistanceMetric':
        """Przełącza metrykę porównywania twarzy i aktualizuje tolerancję."""
        new_metric = self.face_recognition.toggle_metric()
        self._update_face_recognition_tolerance()  # Zaktualizuj tolerancję dla nowej metryki
        logger.info(f"Zmieniono metrykę na: {new_metric.value}")
        return new_metric

    def get_emotion_summary(self, user_id: str) -> Dict[str, float]:
        """Zwraca podsumowanie emocji dla danego użytkownika."""
        if user_id not in self.sessions:
            return {}
        session = self.sessions[user_id]
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
        self._update_fps()
        """Rysuje interfejs użytkownika na ramce wideo."""
        img = frame.copy()
        height, width = img.shape[:2]

        # Tło dla paska statusu
        cv2.rectangle(img, (0, 0), (width, 40), (50, 50, 50), -1)

        # Dynamiczne pozycjonowanie
        x_offset = 10
        y_pos = 25

        # FPS
        fps_text = f"FPS: {self.fps:.1f}"
        _draw_text(img, fps_text, (x_offset, y_pos), 0.7, (0, 255, 0), 2)
        (fps_w, _), _ = cv2.getTextSize(fps_text.encode('ascii', 'ignore').decode('ascii'), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x_offset += fps_w + 30

        if session:
            status_text = f"Status: {session.auth_state.value}"
            status_color = (0, 255, 0) if session.auth_state == AuthState.AUTHENTICATED else (0, 0, 255)
            _draw_text(img, status_text, (x_offset, y_pos), 0.7, status_color, 2)
            (status_w, _), _ = cv2.getTextSize(status_text.encode('ascii', 'ignore').decode('ascii'), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x_offset += status_w + 30

            user_text = f"Użytkownik: {session.user_id}"
            _draw_text(img, user_text, (x_offset, y_pos), 0.7, (255, 255, 255), 2)
            (user_w, _), _ = cv2.getTextSize(user_text.encode('ascii', 'ignore').decode('ascii'), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x_offset += user_w + 30
            
            # Wyświetlanie pewności, jeśli użytkownik jest uwierzytelniony
            if session.auth_state == AuthState.AUTHENTICATED and session.user_id != "Nieznany":
                confidence_text = f"Pewnosc: {session.confidence:.2f}"
                _draw_text(img, confidence_text, (x_offset, y_pos), 0.7, (0, 255, 255), 2)
                (conf_w, _), _ = cv2.getTextSize(confidence_text.encode('ascii', 'ignore').decode('ascii'), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                x_offset += conf_w + 30

            # Jeśli wykryto emocje, wyświetl dominującą (po prawej stronie)
            if session.last_emotions:
                emotion = session.last_emotions[0].emotion
                emotion_confidence = session.last_emotions[0].confidence
                emotion_text = f"{emotion.value}: {emotion_confidence:.1%}"
                emotion_colors = self.emotion_analyzer.get_emotion_colors()
                color = emotion_colors.get(emotion, (255, 255, 255))
                text_size, _ = cv2.getTextSize(emotion_text.encode('ascii', 'ignore').decode('ascii'), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                _draw_text(img, emotion_text, (width - text_size[0] - 10, y_pos), 0.7, color, 2)

        # Wyświetlanie aktualnej metryki
        metric_text = f"Metryka: {self.face_recognition.metric.value.capitalize()} (M)"
        _draw_text(img, metric_text, (10, height - 15), 0.6, (255, 255, 0), 2)

        return img
    
    def cleanup(self) -> None:
        """Zwalnia zasoby systemu."""
        self._save_sessions()
        cv2.destroyAllWindows()
