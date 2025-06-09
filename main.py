import cv2
import numpy as np
import sys
import logging
from pathlib import Path
import argparse
from typing import Optional, Dict, Any

from config import LOG_CONFIG, CAMERA_SETTINGS
from biometric_system import BiometricSystem, UserSession, AuthState
from face_utils import FaceRecognition
import logging.config

# Konfiguracja logowania
logging.config.dictConfig(LOG_CONFIG)
logger = logging.getLogger(__name__)

# Parametry optymalizacji wydajności
SKIP_FRAMES = 4        # wykonuj detekcję co 5. klatkę
SCALE_FACTOR = 0.5     # skala obrazu przy detekcji

class BiometricApp:
    def __init__(self, camera_id: int = 0):
        """Inicjalizacja aplikacji biometrycznej.
        
        Args:
            camera_id: ID kamery do użycia
        """
        self.camera_id = camera_id
        self.cap = None
        self.biometric_system = BiometricSystem()
        self.face_recognition = FaceRecognition()
        self.running = False
        self.current_user: Optional[UserSession] = None
        # Parametry wydajności
        self.skip_frames = SKIP_FRAMES
        self.scale_factor = SCALE_FACTOR
        self.frame_idx = 0
        self.face_results_for_drawing = []  # Przechowuje (name, conf, scaled_location) do rysowania na pełnej klatce
        
        # Inicjalizacja interfejsu użytkownika
        self.window_name = "System Biometryczny"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
    
    def initialize_camera(self) -> bool:
        """Inicjalizuje kamerę.
        
        Returns:
            bool: True jeśli inicjalizacja się powiodła, False w przeciwnym wypadku
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Nie można otworzyć kamery o ID: {self.camera_id}")
                return False
            
            # Ustawienie rozdzielczości kamery
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
            
            logger.info(f"Kamera zainicjalizowana. Rozdzielczość: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas inicjalizacji kamery: {e}")
            return False
    
    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        # Import _draw_text once if needed for this method, or ensure it's available
        # For drawing face boxes, it's done after biometric_system.draw_ui
        try:
            _draw_text_fn = self.biometric_system._draw_text # Access via instance
        except AttributeError:
            # Fallback or log if _draw_text is not found, though it should be part of BiometricSystem
            _draw_text_fn = lambda img, text, pos, scale, color, thick: cv2.putText(img, text.encode('ascii', 'ignore').decode('ascii'), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

        """Przetwarza pojedynczą klatkę wideo.
        
        Args:
            frame: Klatka wideo do przetworzenia
            
        Returns:
            Przetworzona klatka
        """
        try:
            if self.frame_idx % (self.skip_frames + 1) == 0:
                small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
                # Rozpoznaj twarze na małej klatce RAZ
                face_results_small = self.face_recognition.recognize_face(small_frame)

                # Przekaż wyniki do systemu biometrycznego (który nie będzie już sam wywoływał recognize_face)
                self.current_user = self.biometric_system.authenticate_user_with_results(small_frame, face_results_small)

                # Przygotuj wyniki do rysowania na pełnej klatce
                self.face_results_for_drawing = []
                if face_results_small:
                    for name, conf, (top_s, right_s, bottom_s, left_s) in face_results_small:
                        scale_inv = 1.0 / self.scale_factor
                        top = int(top_s * scale_inv)
                        right = int(right_s * scale_inv)
                        bottom = int(bottom_s * scale_inv)
                        left = int(left_s * scale_inv)
                        self.face_results_for_drawing.append((name, conf, (top, right, bottom, left)))
            self.frame_idx += 1

            # Narysuj główny interfejs użytkownika (pasek statusu, FPS itp.)
            # Ta metoda NIE powinna już rysować ramek wokół twarzy, to zrobimy poniżej
            processed_frame = self.biometric_system.draw_ui(frame.copy(), self.current_user) # Działaj na kopii

            # Rysuj prostokąty oraz etykiety twarzy na klatce z UI
            if self.face_results_for_drawing:
                for name, conf, (top, right, bottom, left) in self.face_results_for_drawing:
                    color = (0, 255, 0) if name != "Nieznany" else (0, 0, 255)
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
                    # Użyj _draw_text_fn zdefiniowanego na początku metody
                    _draw_text_fn(processed_frame, name, (left, max(20, top - 10)), 0.7, color, 2)
            frame = processed_frame # Zaktualizuj oryginalną klatkę
            
            return frame
            
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania klatki: {e}", exc_info=True)
            return frame
    
    def register_user(self) -> None:
        """Rejestruje nowego użytkownika."""
        try:
            # Dostęp do _draw_text przez instancję BiometricSystem, jeśli jest tam jako metoda lub atrybut publiczny
            # lub zaimportuj bezpośrednio, jeśli jest to funkcja na poziomie modułu biometric_system
            _draw_text_fn = self.biometric_system._draw_text 
        except AttributeError:
            # Fallback, jeśli _draw_text nie jest dostępny przez self.biometric_system
            # To sugeruje, że _draw_text powinien być importowany inaczej lub udostępniony
            import importlib
            biometric_system_module = importlib.import_module('biometric_system')
            _draw_text_fn = biometric_system_module._draw_text

        
        if not self.cap or not self.cap.isOpened():
            logger.error("Kamera nie jest dostępna do rejestracji.")
            return
        
        # Tworzymy okno do wprowadzania tekstu
        user_id = ""
        input_active = True
        
        try:
            while input_active:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Nie można odczytać klatki z kamery.")
                    break

                display_frame = frame.copy()
                small_frame = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
                
                # Użyj recognize_face na małej klatce do feedbacku UI
                # To jest wolniejsze niż face_locations, ale daje spójność z główną pętlą
                face_results_small = self.face_recognition.recognize_face(small_frame)
                
                face_detected_on_small = False # Resetuj flagę dla każdej klatki
                if face_results_small:
                    face_detected_on_small = True
                    for _name, _conf, (top_s, right_s, bottom_s, left_s) in face_results_small:
                        # Skaluj koordynaty do rysowania na display_frame (pełna rozdzielczość)
                        scale_inv = 1.0 / self.scale_factor
                        top = int(top_s * scale_inv)
                        right = int(right_s * scale_inv)
                        bottom = int(bottom_s * scale_inv)
                        left = int(left_s * scale_inv)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        # Można dodać etykietę tymczasową, np. "Potencjalna twarz"
                        # _draw_text_fn(display_frame, "Twarz?", (left, max(20, top - 10)), 0.6, (0,255,0), 1)
                
                # Rysujemy interfejs na display_frame
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 110), (50, 50, 50), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                _draw_text_fn(display_frame, "Wprowadź identyfikator użytkownika:", (10, 30), 0.7, (255, 255, 255), 2)
                _draw_text_fn(display_frame, user_id, (10, 70), 1.0, (0, 255, 0), 2)
                _draw_text_fn(display_frame, "Enter — zatwierdź, ESC — anuluj", (10, 100), 0.55, (200, 200, 200), 1)
                
                cv2.imshow(self.window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 13:  # Enter
                    if user_id.strip():
                        input_active = False
                        break
                elif key == 27:  # ESC
                    logger.info("Anulowano wprowadzanie identyfikatora.")
                    return
                elif key == 8:  # Backspace
                    user_id = user_id[:-1]
                elif 32 <= key <= 126:  # Znaki drukowalne
                    user_id += chr(key)
        
        except Exception as e:
            logger.error(f"Błąd podczas wprowadzania identyfikatora: {e}")
            return
        
        # Po wprowadzeniu identyfikatora sprawdź unikalność
        if not input_active and user_id.strip() and user_id in self.face_recognition.get_registered_users(): # Sprawdź po zakończeniu wpisywania
            logger.warning(f"Użytkownik '{user_id}' już istnieje w bazie. Anulowanie rejestracji.")
            # Potrzebujemy klatki do wyświetlenia komunikatu
            temp_frame_for_msg = self.cap.read()[1] if self.cap and self.cap.isOpened() else np.zeros((480, 640, 3), dtype=np.uint8)
            if temp_frame_for_msg is not None:
                _draw_text_fn(temp_frame_for_msg, "Użytkownik już istnieje!", (10, 150), 0.8, (0,0,255), 2)
                cv2.imshow(self.window_name, temp_frame_for_msg)
                cv2.waitKey(2000)
            return
        
        # Teraz przechodzimy do rejestracji twarzy
        logger.info(f"Rozpoczęcie rejestracji użytkownika: {user_id}")
        
        face_detected_on_small = False
        registration_capture_active = True
        try:
            while registration_capture_active:
                ret, frame_full_res = self.cap.read()
                if not ret:
                    logger.error("Nie można odczytać klatki z kamery podczas rejestracji.")
                    break

                display_frame = frame_full_res.copy()
                small_frame = cv2.resize(frame_full_res, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
                
                # Użyj recognize_face na małej klatce do feedbacku UI
                # To jest wolniejsze niż face_locations, ale daje spójność z główną pętlą
                face_results_small = self.face_recognition.recognize_face(small_frame)
                
                face_detected_on_small = False # Resetuj flagę dla każdej klatki
                if face_results_small:
                    face_detected_on_small = True
                    for _name, _conf, (top_s, right_s, bottom_s, left_s) in face_results_small:
                        # Skaluj koordynaty do rysowania na display_frame (pełna rozdzielczość)
                        scale_inv = 1.0 / self.scale_factor
                        top = int(top_s * scale_inv)
                        right = int(right_s * scale_inv)
                        bottom = int(bottom_s * scale_inv)
                        left = int(left_s * scale_inv)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        # Można dodać etykietę tymczasową, np. "Potencjalna twarz"
                        # _draw_text_fn(display_frame, "Twarz?", (left, max(20, top - 10)), 0.6, (0,255,0), 1)
                
                # Rysujemy interfejs na display_frame
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 110), (50, 50, 50), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
                
                _draw_text_fn(display_frame, f"Rejestracja: {user_id}", (10, 30), 0.7, (255, 255, 255), 2)
                
                if face_detected_on_small:
                    _draw_text_fn(display_frame, "Twarz wykryta! Naciśnij SPACJĘ, aby zrobić zdjęcie", (10, 70), 0.65, (0, 255, 0), 2)
                else:
                    _draw_text_fn(display_frame, "Zbliż się do kamery i upewnij się, że Twoja twarz jest widoczna", (10, 70), 0.65, (0, 0, 255), 2)
                
                _draw_text_fn(display_frame, "ESC — anuluj", (10, 100), 0.55, (200, 200, 200), 1)
                
                cv2.imshow(self.window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 32 and face_detected_on_small:  # SPACJA -> zapis zdjęcia
                    # Użyj frame_full_res do rejestracji dla lepszej jakości
                    registered = self.face_recognition.register_face(frame_full_res, user_id)
                    if registered:
                        _draw_text_fn(display_frame, "Zarejestrowano pomyślnie!", (10, 140), 0.8, (0,255,0), 2)
                        cv2.imshow(self.window_name, display_frame)
                        cv2.waitKey(2000)
                        registration_capture_active = False # Zakończ pętlę rejestracji
                    else:
                        # Komunikat o nieudanej rejestracji (np. duplikat wg. face_utils lub inny błąd)
                        _draw_text_fn(display_frame, "Rejestracja nieudana.", (10, 140), 0.8, (0,0,255), 2)
                        cv2.imshow(self.window_name, display_frame)
                        cv2.waitKey(2000)
                        registration_capture_active = False # Zakończ pętlę rejestracji
                
                elif key == 27:  # ESC
                    logger.info("Anulowano rejestrację przez użytkownika.")
                    registration_capture_active = False # Zakończ pętlę rejestracji

                    
        except Exception as e:
            logger.error(f"Błąd podczas rejestracji użytkownika: {e}", exc_info=True)
    
    def list_users(self) -> None:
        """Wyświetla listę zarejestrowanych użytkowników w osobnym oknie OpenCV."""
        import importlib
        biometric_system = importlib.import_module('biometric_system')
        _draw_text = biometric_system._draw_text
        users = sorted(self.face_recognition.get_registered_users())
        showing_list = True
        win_name = "Użytkownicy"

        # Okno o stałym rozmiarze
        width, height = 400, 600
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, width, height)

        while showing_list:
            # Czarne tło
            display_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Panel tytułu
            cv2.rectangle(display_frame, (0, 0), (width, 60), (60, 10, 10), -1)
            _draw_text(display_frame, "Zarejestrowani użytkownicy:", (15, 40), 0.9, (255,255,255), 2)
            cv2.line(display_frame, (10, 55), (width-10, 55), (110, 40, 40), 1)

            # Lista użytkowników
            if not users:
                _draw_text(display_frame, "Brak zarejestrowanych użytkowników", (15, 100), 0.7, (200,200,200), 1)
            else:
                max_users_visible = (height-150)//32
                start_idx = 0
                for i in range(start_idx, min(start_idx+max_users_visible, len(users))):
                    y_pos = 100 + (i-start_idx)*32
                    user_text = f"{i+1}. {users[i]}"
                    # Pasek podświetlenia dla pierwszego (możesz rozbudować o wybór)
                    if i == 0:
                        cv2.rectangle(display_frame, (8, y_pos-22), (width-8, y_pos+8), (80,50,50), -1)
                    _draw_text(display_frame, user_text, (20, y_pos), 0.75, (240,240,255), 2)
            # Stopka
            cv2.rectangle(display_frame, (0, height-50), (width, height), (40,40,60), -1)
            _draw_text(display_frame, f"Liczba użytkowników: {len(users)}", (15, height-20), 0.65, (180,180,255), 1)
            _draw_text(display_frame, "ESC - powrót", (width-140, height-20), 0.6, (150,150,255), 1)

            cv2.imshow(win_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                showing_list = False
                cv2.destroyWindow(win_name)
                break
    
    def run(self) -> None:
        """Uruchamia główną pętlę aplikacji."""
        if not self.initialize_camera():
            logger.error("Nie można uruchomić aplikacji z powodu błędu kamery.")
            return
        
        self.running = True
        logger.info("Aplikacja uruchomiona. Naciśnij 'q', aby zakończyć.")
        
        try:
            while self.running:
                # Odczytaj klatkę z kamery
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Nie można odczytać klatki z kamery.")
                    break
                
                # Przetwórz klatkę
                processed_frame = self.process_frame(frame)
                
                # Wyświetl wynik
                cv2.imshow(self.window_name, processed_frame)
                
                # Obsługa klawiszy
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Wyjście
                    self.running = False # Ustaw flagę, aby wyjść z pętli
                elif key == ord('r'):  # Rejestracja nowego użytkownika
                    self.register_user()
                elif key == ord('l'):  # Lista zarejestrowanych użytkowników
                    self.list_users()
                
        except KeyboardInterrupt:
            logger.info("Zatrzymywanie aplikacji przez KeyboardInterrupt...")
            self.running = False # Upewnij się, że pętla się zakończy
        except Exception as e:
            logger.error(f"Wystąpił błąd w pętli głównej: {e}", exc_info=True)
            self.running = False # Upewnij się, że pętla się zakończy
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Zwalnia zasoby."""
        self.running = False
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        self.biometric_system.cleanup()
        cv2.destroyAllWindows()
        logger.info("Aplikacja zakończyła działanie.")

def parse_arguments():
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description='System biometryczny z rozpoznawaniem twarzy i emocji.')
    parser.add_argument('--camera', type=int, default=0, 
                       help='ID kamery do użycia (domyślnie: 0)')
    return parser.parse_args()

def main():
    """Główna funkcja aplikacji."""
    args = parse_arguments()
    
    try:
        app = BiometricApp(camera_id=args.camera)
        app.run()
    except Exception as e:
        logger.critical(f"Krytyczny błąd: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
