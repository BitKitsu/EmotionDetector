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

logger.info("Using GTK3 text rendering with native UTF-8 support.")


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
    emotion_baseline: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje obiekt sesji na słownik."""
        return {
            'user_id': self.user_id,
            'auth_state': self.auth_state.value,
            'last_seen': self.last_seen,
            'confidence': self.confidence,
            'auth_attempts': self.auth_attempts,
            'session_start': self.session_start,
            'emotion_baseline': self.emotion_baseline,
            'emotion_history': [
                 { (e_type.name if hasattr(e_type, 'name') else str(e_type)): score
                   for e_type, score in emotions.items() }
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
        session.emotion_baseline = data.get('emotion_baseline')

        
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

        # --- Śledzenie twarzy (optymalizacja wydajności) ---
        self.active_trackers: Dict[int, Tuple[Any, str, float]] = {}  # tracker_id -> (tracker_obj, user_id, confidence)
        self.next_tracker_id = 0
        self.frames_since_last_full_recognition = 0
        self.recognition_interval = 30  # Uruchom pełne rozpoznawanie co 30 klatek
    
    def _load_sessions(self) -> None:
        """Wczytuje zapisane sesje użytkowników."""
        sessions_file = DATA_DIR / 'user_sessions.json'
        if sessions_file.exists() and sessions_file.stat().st_size > 0:
            try:
                with open(sessions_file, 'r') as f:
                    sessions_data = json.load(f)
                    self.sessions = {
                        user_id: UserSession.from_dict(session_data)
                        for user_id, session_data in sessions_data.items()
                    }
                logger.info(f"Wczytano {len(self.sessions)} sesji użytkowników")
            except json.JSONDecodeError as e:
                logger.error(f"Błąd podczas wczytywania sesji: {e}")
                self.sessions = {}
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

        # Analiza emocji
        emotion_results, new_baseline = self.emotion_analyzer.detect_emotions(frame, current_user_id=user_id)

        # Jeśli kalibracja się zakończyła, zapisz nową linię bazową w sesji
        if new_baseline:
            session.emotion_baseline = new_baseline
            self._save_sessions()  # Zapisz sesje po udanej kalibracji

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

                    # Po udanym uwierzytelnieniu, załaduj zapisaną linię bazową (jeśli istnieje)
                    if session.emotion_baseline:
                        self.emotion_analyzer.set_baseline(session.emotion_baseline)
                    else:
                        # Jeśli użytkownik nie ma linii bazowej, zresetuj do domyślnej
                        self.emotion_analyzer.baseline_raw = None
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

    def authenticate_user(self, frame: np.ndarray) -> Optional[UserSession]:
        """Uwierzytelnia użytkownika, używając optymalizacji śledzenia."""
        self._update_fps()
        self.frames_since_last_full_recognition += 1
        face_results = []

        # Najpierw spróbuj zaktualizować istniejące trackery
        if self.active_trackers:
            face_results = self._update_trackers(frame)

        # Jeśli nie ma śledzonych twarzy lub nadszedł czas na pełne skanowanie
        if not self.active_trackers or self.frames_since_last_full_recognition >= self.recognition_interval:
            self.frames_since_last_full_recognition = 0
            # Optymalizacja: skalowanie klatki przed detekcją
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            face_results = self._run_full_recognition(frame, small_frame)

        # Dalsza logika pozostaje taka sama
        session = self.authenticate_user_with_results(frame, face_results)
        if session:
            session.face_results = face_results
        return session

    def _update_trackers(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple]]:
        """Aktualizuje pozycje aktywnych trackerów i zwraca wyniki."""
        if not self.active_trackers:
            return []

        face_results = []
        failed_tracker_ids = []

        for tracker_id, (tracker, user_id, confidence) in self.active_trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                top, right, bottom, left = y, x + w, y + h, x
                face_results.append((user_id, confidence, (top, right, bottom, left)))
            else:
                logger.warning(f"Tracker ID {tracker_id} dla {user_id} zgubił cel.")
                failed_tracker_ids.append(tracker_id)

        for tracker_id in failed_tracker_ids:
            del self.active_trackers[tracker_id]

        return face_results

    def _run_full_recognition(self, frame: np.ndarray, small_frame: np.ndarray) -> List[Tuple[str, float, Tuple]]:
        """Przeprowadza pełny, kosztowny proces rozpoznawania i inicjalizuje nowe trackery."""
        logger.debug("Uruchamianie pełnego rozpoznawania twarzy...")
        recognized_faces_small = self.face_recognition.recognize_face(small_frame)
        
        self.active_trackers.clear()
        self.next_tracker_id = 0

        new_face_results = []
        for user_id, confidence, (top_s, right_s, bottom_s, left_s) in recognized_faces_small:
            top = int(top_s * 2)
            right = int(right_s * 2)
            bottom = int(bottom_s * 2)
            left = int(left_s * 2)
            
            x, y, w, h = left, top, right - left, bottom - top
            bbox = (x, y, w, h)

            try:
                try:
                    tracker = cv2.TrackerCSRT_create()
                except AttributeError:
                    logger.warning("Tracker CSRT niedostępny, używam MOSSE...")
                    tracker = cv2.legacy.TrackerMOSSE_create()

                tracker.init(frame, bbox)

                tracker_id = self.next_tracker_id
                self.active_trackers[tracker_id] = (tracker, user_id, confidence)
                self.next_tracker_id += 1
                
                new_face_results.append((user_id, confidence, (top, right, bottom, left)))
                logger.debug(f"Zainicjalizowano nowy tracker ID {tracker_id} dla {user_id}")
            except Exception as e:
                logger.error(f"Nie udało się zainicjalizować trackera dla {user_id}: {e}")
                
        return new_face_results

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
        """Rysuje interfejs użytkownika na ramce wideo."""
        self._update_fps()
        img = frame.copy()
        height, width, _ = img.shape

        # Tło dla paska statusu
        cv2.rectangle(img, (0, 0), (width, 40), (20, 20, 20), -1)

        # Wyświetlanie FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(img, fps_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        if session and session.face_results:
            # Rysowanie informacji o sesji i twarzach
            for user_id, confidence, (x, y, w, h) in session.face_results:
                # Ramka wokół twarzy
                color = (0, 255, 0) if session.auth_state == AuthState.AUTHENTICATED else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                # Tekst z ID użytkownika i statusem
                auth_text = f"{user_id} ({session.auth_state.value})"
                cv2.putText(img, auth_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                # Pewność dopasowania
                conf_text = f"Pewnosc: {confidence:.2f}"
                cv2.putText(img, conf_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                # Wyświetlanie emocji tylko dla zarejestrowanych użytkowników
                if user_id != "Nieznany":
                    if session.last_emotions:
                        for i, emotion_res in enumerate(session.last_emotions[:3]): # Pokaż do 3 emocji
                            emotion_text = f"{emotion_res.emotion.value}: {emotion_res.confidence:.1%}"
                            emotion_colors = self.emotion_analyzer.get_emotion_colors()
                            emotion_color = emotion_colors.get(emotion_res.emotion, (255, 255, 255))
                            cv2.putText(img, emotion_text, (x, y + h + 50 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2, cv2.LINE_AA)

        # Wyświetlanie aktualnej metryki na dole
        metric_text = f"Metryka: {self.face_recognition.metric.value.capitalize()} (M)"
        cv2.putText(img, metric_text, (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        return img
    
    def cleanup(self) -> None:
        """Zwalnia zasoby systemu."""
        self._save_sessions()
        cv2.destroyAllWindows()
