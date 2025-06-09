import cv2
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
        """Przetwarza pojedynczą klatkę wideo.
        
        Args:
            frame: Klatka wideo do przetworzenia
            
        Returns:
            Przetworzona klatka
        """
        try:
            # Wykonaj uwierzytelnianie biometryczne
            self.current_user = self.biometric_system.authenticate_user(frame)
            
            # Narysuj interfejs użytkownika
            frame = self.biometric_system.draw_ui(frame, self.current_user)
            
            return frame
            
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania klatki: {e}", exc_info=True)
            return frame
    
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
                    break
                elif key == ord('r'):  # Rejestracja nowego użytkownika
                    self.register_user()
                elif key == ord('l'):  # Lista zarejestrowanych użytkowników
                    self.list_users()
                
        except KeyboardInterrupt:
            logger.info("Zatrzymywanie aplikacji...")
        except Exception as e:
            logger.error(f"Krytyczny błąd: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def register_user(self) -> None:
        """Rejestruje nowego użytkownika."""
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
                
                # Rysujemy interfejs wprowadzania tekstu
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (50, 50, 50), -1)
                alpha = 0.6  # Przezroczystość
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Wyświetlamy instrukcje
                cv2.putText(frame, "Wprowadź identyfikator użytkownika:", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, user_id, (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, "Enter - zatwierdź, ESC - anuluj", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Pokazujemy podgląd z kamerki
                cv2.imshow(self.window_name, frame)
                
                # Obsługa klawiszy
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
        
        # Teraz przechodzimy do rejestracji twarzy
        logger.info(f"Rozpoczęcie rejestracji użytkownika: {user_id}")
        
        face_detected = False
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Nie można odczytać klatki z kamery.")
                    break
                
                # Sprawdzamy czy jest twarz na zdjęciu
                face_locations = self.face_recognition.recognize_face(frame)
                
                # Rysujemy prostokąt wokół twarzy jeśli wykryta
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Przygotowanie tekstu
                    label = f"{user_id}"
                    
                    # Obliczenie rozmiaru tekstu
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 0.7
                    thickness = 1
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Rysowanie tła dla tekstu
                    cv2.rectangle(
                        frame, 
                        (left, bottom - 35), 
                        (left + text_width + 12, bottom), 
                        (0, 255, 0), 
                        cv2.FILLED
                    )
                    
                    # Rysowanie tekstu
                    cv2.putText(
                        frame, 
                        label, 
                        (left + 6, bottom - 10), 
                        font, 
                        font_scale, 
                        (255, 255, 255), 
                        thickness,
                        cv2.LINE_AA
                    )
                    
                    face_detected = True
                else:
                    face_detected = False
                
                # Rysujemy interfejs
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (50, 50, 50), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                cv2.putText(frame, f"Rejestracja: {user_id}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if face_detected:
                    cv2.putText(frame, "Twarz wykryta! Naciśnij SPACJĘ, aby zrobić zdjęcie", (10, 70), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Zbliż się do kamery i upewnij się, że Twoja twarz jest widoczna", 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, "ESC - anuluj", (10, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow(self.window_name, frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    logger.info("Anulowano rejestrację.")
                    return
                elif key == 32 and face_detected:  # Spacja
                    # Zarejestruj użytkownika
                    success = self.biometric_system.register_user(user_id, frame)
                    if success:
                        logger.info(f"Pomyślnie zarejestrowano użytkownika: {user_id}")
                        # Pokaż komunikat o sukcesie
                        cv2.putText(frame, "Zarejestrowano pomyślnie!", 
                                  (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.imshow(self.window_name, frame)
                        cv2.waitKey(2000)  # Pokazujemy komunikat przez 2 sekundy
                    else:
                        logger.error(f"Nie udało się zarejestrować użytkownika: {user_id}")
                        cv2.putText(frame, "Błąd rejestracji! Spróbuj ponownie.", 
                                  (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow(self.window_name, frame)
                        cv2.waitKey(2000)  # Pokazujemy komunikat przez 2 sekundy
                    break
                    
        except Exception as e:
            logger.error(f"Błąd podczas rejestracji użytkownika: {e}", exc_info=True)
    
    def list_users(self) -> None:
        """Wyświetla listę zarejestrowanych użytkowników w interfejsie programu."""
        users = sorted(self.face_recognition.get_registered_users())
        showing_list = True
        
        while showing_list and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Nie można odczytać klatki z kamery.")
                time.sleep(0.1)
                continue
            
            # Tworzymy kopię klatki do modyfikacji
            frame_copy = frame.copy()
            
            # Obliczamy szerokość panelu bocznego (30% szerokości klatki)
            panel_width = int(frame.shape[1] * 0.3)
            
            # Rysujemy półprzezroczyste tło dla panelu bocznego
            overlay = frame_copy.copy()
            cv2.rectangle(overlay, (0, 0), (panel_width, frame.shape[0]), (40, 40, 60), -1)
            alpha = 0.9  # Mocniejsze tło dla lepszej czytelności
            cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0, frame_copy)
            
            # Rysujemy obramowanie panelu
            cv2.rectangle(frame_copy, (0, 0), (panel_width, frame.shape[0]), (80, 80, 255), 2)
            
            # Dodajemy nagłówek
            cv2.putText(frame_copy, "Zarejestrowani użytkownicy:", 
                       (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, 
                       (255, 255, 255), 
                       2)
            
            # Rysujemy linię pod nagłówkiem
            cv2.line(frame_copy, (10, 50), (panel_width - 10, 50), (100, 100, 255), 1)
            
            # Wyświetlamy listę użytkowników
            if not users:
                cv2.putText(frame_copy, "Brak zarejestrowanych użytkowników", 
                           (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, 
                           (200, 200, 200), 
                           1)
            else:
                max_users_visible = (frame.shape[0] - 150) // 30  # Maksymalna liczba użytkowników widocznych na raz
                
                # Obliczamy, od którego użytkownika zacząć wyświetlanie (do przewijania)
                start_idx = max(0, min(len(users) - max_users_visible, 0))
                
                for i in range(start_idx, min(start_idx + max_users_visible, len(users))):
                    y_pos = 90 + (i - start_idx) * 30
                    user_text = f"{i+1}. {users[i]}"
                    
                    # Rysujemy tło dla aktywnego użytkownika
                    if i == 0:  # Można dodać logikę zaznaczania aktywnego użytkownika
                        cv2.rectangle(frame_copy, 
                                    (5, y_pos - 20), 
                                    (panel_width - 5, y_pos + 5), 
                                    (60, 60, 80), 
                                    -1)
                    
                    cv2.putText(frame_copy, user_text, 
                               (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, 
                               (200, 200, 255) if i % 2 == 0 else (180, 220, 255), 
                               1)
            
            # Dodajemy stopkę z informacją o liczbie użytkowników
            footer_bg = frame_copy[frame.shape[0]-40:frame.shape[0], 0:panel_width]
            cv2.rectangle(footer_bg, (0, 0), (panel_width, 40), (30, 30, 50), -1)
            frame_copy[frame.shape[0]-40:frame.shape[0], 0:panel_width] = footer_bg
            
            cv2.putText(frame_copy, f"Liczba użytkowników: {len(users)}", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (180, 180, 255), 
                       1)
            
            # Dodajemy instrukcję
            cv2.putText(frame_copy, "ESC - powrót", 
                       (panel_width - 120, frame.shape[0] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (150, 150, 255), 
                       1)
            
            # Wyświetlamy klatkę
            cv2.imshow(self.window_name, frame_copy)
            
            # Sprawdzamy naciśnięcie klawisza
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                showing_list = False
                # Oczyszczamy ekran przed powrotem
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
    
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
