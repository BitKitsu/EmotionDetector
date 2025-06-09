import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any

class EmotionDetector:
    def _count_cameras(self) -> int:
        """Zlicza dostępne kamery."""
        max_tested = 10
        available = 0
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available += 1
                cap.release()
        return available
        
    def __init__(self, camera_id: int = 0, update_interval: float = 0.1):
        print(f"Inicjalizacja kamery o ID: {camera_id}")
        self.cap = cv2.VideoCapture(camera_id)
        # Ustawienie rozdzielczości na standardową
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.update_interval = update_interval
        self.last_update = 0
        self.prev_frame_time = time.time()
        
        # Inicjalizacja Mediapipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Punkty landmarków
        self.LIPS_LEFT = 61
        self.LIPS_RIGHT = 291
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.BROW_LEFT = 70
        self.BROW_RIGHT = 300
        self.EYE_LEFT = 33
        self.EYE_RIGHT = 263
        
        # Inicjalizacja wykresu i podglądu kamery
        plt.ion()
        # Utworzenie figury i siatki
        self.fig = plt.figure(figsize=(12, 5))
        gs = self.fig.add_gridspec(1, 2, wspace=0.3)

        # Oś radarowa (pierwsza kolumna, układ polarny)
        self.ax_radar = self.fig.add_subplot(gs[0, 0], polar=True)

        # Oś podglądu kamery (druga kolumna)
        self.ax_cam = self.fig.add_subplot(gs[0, 1])
        self.ax_cam.axis('off')

        # Etykiety i inicjalizacja danych wykresu
        self.labels = ['Uśmiech', 'Zdziwienie', 'Złość', 'Smutek']
        self.radar_angles = np.linspace(0, 2 * np.pi, len(self.labels) + 1)
        self.radar_values = [0] * (len(self.labels) + 1)
        (self.radar_line,) = self.ax_radar.plot(self.radar_angles, self.radar_values, 'r-', linewidth=2)
        self.ax_radar.set_thetagrids(np.degrees(self.radar_angles[:-1]), self.labels)
        self.ax_radar.set_ylim(0, 0.2)

        # Inicjalizacja obrazu kamery
        self.img_display = self.ax_cam.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
    def get_emotion_values(self, landmarks, w: int, h: int) -> Dict[str, float]:
        """Oblicza wartości emocji na podstawie punktów charakterystycznych twarzy."""
        def p(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        try:
            mouth_width = np.linalg.norm(p(self.LIPS_RIGHT) - p(self.LIPS_LEFT))
            mouth_open = np.linalg.norm(p(self.MOUTH_BOTTOM) - p(self.MOUTH_TOP))
            brow_height = (p(self.BROW_LEFT)[1] + p(self.BROW_RIGHT)[1]) / 2
            eye_height = (p(self.EYE_LEFT)[1] + p(self.EYE_RIGHT)[1]) / 2
            brow_eye_dist = eye_height - brow_height

            # Normalizacja
            smile = np.clip(mouth_width / w, 0, 0.2)
            surprise = np.clip(mouth_open / h, 0, 0.2)
            anger = np.clip(0.05 - brow_eye_dist / h, 0, 0.2)
            sadness = np.clip(0.1 - smile, 0, 0.2)

            return {
                'Uśmiech': smile,
                'Zdziwienie': surprise,
                'Złość': anger,
                'Smutek': sadness
            }
        except Exception as e:
            print(f"Błąd w obliczaniu emocji: {e}")
            return {label: 0 for label in self.labels}
    
    def update_radar(self, emotions: Dict[str, float], frame: np.ndarray = None) -> None:
        """Aktualizuje wykres radarowy z emocjami i wyświetla obraz z kamery."""
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
            
        self.last_update = current_time
        
        # Aktualizacja wykresu radarowego (tylko dane, bez czyszczenia osi)
        values = [emotions.get(label, 0) for label in self.labels]
        values += values[:1]  # Zamyka wykres
        self.radar_line.set_ydata(values)
        # Jeśli chcesz wypełnienie, można zaktualizować kolekcję, ale pomijamy dla wydajności

        # Aktualizacja obrazu z kamery
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.img_display.set_data(frame_rgb)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run(self) -> None:
        """Główna pętla programu."""
        if not self.cap.isOpened():
            print("Błąd: Nie można otworzyć kamery. Sprawdź, czy kamera jest podłączona i dostępna.")
            print(f"Próba otwarcia kamery o ID: {self.cap.get(cv2.CAP_PROP_HW_DEVICE)}")
            print(f"Liczba dostępnych kamer: {self._count_cameras()}")
            return

        print(f"Kamera otwarta pomyślnie. Rozdzielczość: {int(self.cap.get(3))}x{int(self.cap.get(4))}")
        
        # Testowe przechwycenie klatki
        ret, test_frame = self.cap.read()
        if not ret:
            print("Błąd: Nie można odczytać klatki testowej z kamery.")
            return
            
        print(f"Testowa klatka o rozmiarze: {test_frame.shape}")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Błąd: Nie można odczytać klatki z kamery.")
                    break

                h, w, _ = frame.shape
                # Oblicz FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - self.prev_frame_time) if (curr_time - self.prev_frame_time) > 0 else 0
                self.prev_frame_time = curr_time
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    emotions = self.get_emotion_values(landmarks, w, h)
                    
                    # Rysowanie punktów charakterystycznych na obrazie
                    frame_with_landmarks = frame.copy()
                    for idx, lm in enumerate(landmarks):
                        if idx in [self.LIPS_LEFT, self.LIPS_RIGHT, self.MOUTH_TOP, 
                                 self.MOUTH_BOTTOM, self.BROW_LEFT, self.BROW_RIGHT, 
                                 self.EYE_LEFT, self.EYE_RIGHT]:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame_with_landmarks, (x, y), 3, (0, 255, 0), -1)
                    
                    # Dodanie licznika FPS w lewym górnym rogu
                    cv2.putText(frame_with_landmarks, f"FPS: {fps:.1f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Aktualizacja wykresu i obrazu
                    self.update_radar(emotions, frame_with_landmarks)
                else:
                    # Jeśli nie wykryto twarzy, wyświetl tylko obraz z licznikiem FPS
                    frame_disp = frame.copy()
                    cv2.putText(frame_disp, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.update_radar({}, frame_disp)
                
                # Sprawdź, czy użytkownik chce wyjść (zamknięcie okna)
                if not plt.fignum_exists(self.fig.number):
                    break

        except KeyboardInterrupt:
            print("Zatrzymywanie programu...")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Zwalnia zasoby."""
        self.cap.release()
        cv2.destroyAllWindows()
        plt.close('all')

def main():
    detector = EmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()