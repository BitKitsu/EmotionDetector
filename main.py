import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, GdkPixbuf
import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from threading import Thread, Lock
import sys

from face_utils import FaceRecognition, DistanceMetric
from emotion_analyzer import EmotionAnalyzer
from biometric_system import BiometricSystem, UserSession, AuthState
from config import LOG_CONFIG
import logging.config

# Konfiguracja logowania
logging.config.dictConfig(LOG_CONFIG)

logger = logging.getLogger(__name__)

# --- Wątek do przechwytywania klatek i rozpoznawania twarzy ---
class CaptureThread(Thread):
    """Oddzielny wątek pobierający klatki z kamery i wykonujący rozpoznawanie.

    Dzięki temu główny wątek GTK pozostaje responsywny."""

    def __init__(self, system: 'BiometricSystem', camera_id: int = 0):
        super().__init__(daemon=True)
        self.system = system
        self.camera_id = camera_id
        self.running = True
        self.frame_lock = Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_session: Optional[UserSession] = None

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            # Wykonaj rozpoznawanie w tym wątku
            session = self.system.authenticate_user(frame)
            # Zapisz wyniki w sposób bezpieczny wątkowo
            with self.frame_lock:
                self.latest_frame = frame
                self.latest_session = session
        cap.release()

    def stop(self):
        self.running = False


class BiometricApp(Gtk.Window):
    def __init__(self, camera_id: int = 0, width=1280, height=720):
        super().__init__(title="System biometryczny")
        self.set_default_size(width, height)
        self.camera_id = camera_id
        
        # Inicjalizacja systemu biometrycznego
        self.system = BiometricSystem()
        
        # Główne kontenery
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add(self.main_box)
        
        # Górny pasek z przyciskami
        self.create_header_bar()
        
        # Główny obszar z kamerą i informacjami
        self.content_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.main_box.pack_start(self.content_box, True, True, 0)
        
        # Panel kamery
        self.camera_area = Gtk.DrawingArea()
        self.camera_area.set_size_request(800, 600)
        self.camera_area.connect("draw", self.on_draw_camera)
        self.content_box.pack_start(self.camera_area, True, True, 0)
        self.last_pixbuf = None
        
        # Panel informacyjny
        self.info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        self.info_box.set_size_request(300, -1)
        self.content_box.pack_start(self.info_box, False, False, 0)
        
        # Informacje o użytkowniku
        self.user_info_frame = Gtk.Frame(label="Informacje o użytkowniku")
        self.user_info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5, margin=10)
        self.user_info_frame.add(self.user_info_box)
        self.info_box.pack_start(self.user_info_frame, False, False, 0)
        
        self.status_label = Gtk.Label(label="Status: Niezalogowany")
        self.user_label = Gtk.Label(label="Użytkownik: -")
        self.confidence_label = Gtk.Label(label="Pewność: -")
        self.metric_label = Gtk.Label(label="Metryka: Kosinusowa")
        self.fps_label = Gtk.Label(label="FPS: -")
        
        for widget in [self.status_label, self.user_label, self.confidence_label, self.metric_label, self.fps_label]:
            self.user_info_box.pack_start(widget, False, False, 5)
        
        # Przycisk zmiany metryki
        self.metric_button = Gtk.Button(label="Zmień metrykę na Euklidesową")
        self.metric_button.connect("clicked", self.on_metric_button_clicked)
        self.info_box.pack_start(self.metric_button, False, False, 10)

        # Kontrolki do zmiany progów pewności
        self.confidence_frame = Gtk.Frame(label="Ustawienia progów pewności")
        self.confidence_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5, margin=10)
        self.confidence_frame.add(self.confidence_box)
        self.info_box.pack_start(self.confidence_frame, False, False, 10)

        # Próg pewności dla metryki kosinusowej
        cosine_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        cosine_label = Gtk.Label(label="Pewność (Kosinus):")
        self.cosine_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.01)
        cosine_tolerance = self.system.get_match_threshold(DistanceMetric.COSINE)
        self.cosine_spin.set_value(1.0 - cosine_tolerance)
        self.cosine_spin.connect("value-changed", self._on_confidence_threshold_changed, DistanceMetric.COSINE)
        cosine_hbox.pack_start(cosine_label, False, False, 0)
        cosine_hbox.pack_start(self.cosine_spin, True, True, 0)
        self.confidence_box.pack_start(cosine_hbox, False, False, 0)

        # Próg pewności dla metryki euklidesowej
        euclidean_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=5)
        euclidean_label = Gtk.Label(label="Pewność (Euklides):")
        self.euclidean_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.01)
        euclidean_tolerance = self.system.get_match_threshold(DistanceMetric.EUCLIDEAN)
        self.euclidean_spin.set_value(1.0 - euclidean_tolerance)
        self.euclidean_spin.connect("value-changed", self._on_confidence_threshold_changed, DistanceMetric.EUCLIDEAN)
        euclidean_hbox.pack_start(euclidean_label, False, False, 0)
        euclidean_hbox.pack_start(self.euclidean_spin, True, True, 0)
        self.confidence_box.pack_start(euclidean_hbox, False, False, 0)
        
        # Obszar na emocje
        self.emotion_frame = Gtk.Frame(label="Emocje")
        self.emotion_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5, margin=10)
        self.emotion_frame.add(self.emotion_box)
        self.info_box.pack_start(self.emotion_frame, False, False, 10)
        
        self.emotion_label = Gtk.Label(label="Dominująca emocja: -")
        self.emotion_confidence_label = Gtk.Label(label="Pewność: -")
        
        for widget in [self.emotion_label, self.emotion_confidence_label]:
            self.emotion_box.pack_start(widget, False, False, 5)
        
        # Uruchom wątek kamery
        self.capture_thread = CaptureThread(self.system, self.camera_id)
        self.capture_thread.start()
        
        # Rozpocznij okresowe odświeżanie UI
        self.timeout_id = GLib.timeout_add(30, self.update_ui)  # ~33 FPS
        
        # Połączenie sygnałów
        self.connect("destroy", self.on_destroy)

    def _on_confidence_threshold_changed(self, spin_button, metric):
        """Obsługuje zmianę wartości progu pewności w SpinButton."""
        new_confidence = spin_button.get_value()
        # Konwertuj pewność (0-1) na tolerancję (odległość)
        new_tolerance = 1.0 - new_confidence
        self.system.set_match_threshold(metric, new_tolerance)

    def on_draw_camera(self, widget, cr):
        """Rysuje klatkę z kamery na DrawingArea z zachowaniem proporcji."""
        if self.last_pixbuf is None:
            return

        alloc = widget.get_allocation()
        pixbuf_w = self.last_pixbuf.get_width()
        pixbuf_h = self.last_pixbuf.get_height()

        scale_w = alloc.width / pixbuf_w
        scale_h = alloc.height / pixbuf_h
        scale = min(scale_w, scale_h)

        target_w = int(pixbuf_w * scale)
        target_h = int(pixbuf_h * scale)

        x_offset = (alloc.width - target_w) // 2
        y_offset = (alloc.height - target_h) // 2

        scaled_pixbuf = self.last_pixbuf.scale_simple(
            target_w, target_h, GdkPixbuf.InterpType.BILINEAR
        )

        Gdk.cairo_set_source_pixbuf(cr, scaled_pixbuf, x_offset, y_offset)
        cr.paint()

    def create_header_bar(self):
        header = Gtk.HeaderBar()
        header.set_show_close_button(True)
        header.props.title = "System biometryczny"
        self.set_titlebar(header)
        
        # Przycisk rejestracji
        self.register_button = Gtk.Button(label="Zarejestruj użytkownika")
        self.register_button.connect("clicked", self.on_register_clicked)
        self.register_button.set_sensitive(False)  # Wyłącz na starcie
        header.pack_start(self.register_button)
        
        # Przycisk listy użytkowników
        self.users_btn = Gtk.Button(label="Lista użytkowników")
        self.users_btn.connect("clicked", self.on_users_clicked)
        header.pack_start(self.users_btn)
        
    def on_metric_button_clicked(self, button):
        self.system.toggle_metric()
        current_metric = self.system.face_recognition.metric
        if current_metric == DistanceMetric.EUCLIDEAN:
            self.metric_button.set_label("Zmień metrykę na Kosinusową")
            self.metric_label.set_text("Metryka: Euklidesowa")
        else:
            self.metric_button.set_label("Zmień metrykę na Euklidesową")
            self.metric_label.set_text("Metryka: Kosinusowa")
    
    def update_ui(self):
        # Odczytaj najnowszą klatkę z wątku kamery
        frame = None
        session = None
        with self.capture_thread.frame_lock:
            if self.capture_thread.latest_frame is not None:
                frame = self.capture_thread.latest_frame.copy()
                session = self.capture_thread.latest_session

        if frame is not None:
            # Włącz przycisk rejestracji, gdy kamera jest gotowa
            if not self.register_button.get_sensitive():
                self.register_button.set_sensitive(True)
            # Aktualizacja interfejsu
            if session and session.face_results:
                # Rysowanie ramek i etykiet na klatce
                for name, conf, (top, right, bottom, left) in session.face_results:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = f"{name} ({conf:.2f})"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.status_label.set_text(f"Status: {session.auth_state.value}")
                self.user_label.set_text(f"Użytkownik: {session.user_id}")
                self.confidence_label.set_text(f"Pewność: {session.confidence:.2f}")
                if session.last_emotions:
                    emotion = session.last_emotions[0].emotion
                    confidence = session.last_emotions[0].confidence
                    self.emotion_label.set_text(f"Dominująca emocja: {emotion.value}")
                    self.emotion_confidence_label.set_text(f"Pewność: {confidence:.1%}")
                else:
                    self.emotion_label.set_text("Dominująca emocja: -")
                    self.emotion_confidence_label.set_text("Pewność: -")
            else:
                # Resetuj etykiety, jeśli nie wykryto twarzy
                self.status_label.set_text("Status: Niezalogowany")
                self.user_label.set_text("Użytkownik: -")
                self.confidence_label.set_text("Pewność: -")
                self.emotion_label.set_text("Dominująca emocja: -")
                self.emotion_confidence_label.set_text("Pewność: -")

            # Zawsze aktualizuj FPS
            self.fps_label.set_text(f"FPS: {self.system.fps:.2f}")

            # Konwersja klatki do formatu GTK
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            pixbuf = GdkPixbuf.Pixbuf.new_from_data(
                frame.tobytes(),
                GdkPixbuf.Colorspace.RGB,
                False,
                8,
                width,
                height,
                width * 3,
                None,
                None
            )
            
            # Zapisz pixbuf i zleć odrysowanie
            self.last_pixbuf = pixbuf
            self.camera_area.queue_draw()
        
        # Kontynuuj aktualizację
        return GLib.SOURCE_CONTINUE
    
    def on_register_clicked(self, button):
        dialog = Gtk.Dialog(title="Rejestracja nowego użytkownika", transient_for=self, flags=0)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)

        content_area = dialog.get_content_area()
        label = Gtk.Label(label="Wprowadź nazwę użytkownika poniżej i upewnij się, że Twoja twarz jest dobrze widoczna w kamerze.")
        content_area.add(label)

        entry = Gtk.Entry()
        entry.set_placeholder_text("Nazwa użytkownika")
        content_area.add(entry)
        dialog.show_all()

        response = dialog.run()
        user_id = entry.get_text().strip()
        dialog.destroy()

        if response == Gtk.ResponseType.OK and user_id:
            if user_id in self.system.face_recognition.get_registered_users():
                self.show_message("Błąd", f"Użytkownik '{user_id}' już istnieje.", Gtk.MessageType.ERROR)
                return

            frame = None
            session = None
            with self.capture_thread.frame_lock:
                if self.capture_thread.latest_frame is not None:
                    frame = self.capture_thread.latest_frame.copy()
                    session = self.capture_thread.latest_session

            if frame is None or session is None:
                self.show_message("Błąd", "Brak obrazu z kamery lub danych sesji.", Gtk.MessageType.ERROR)
                return

            # Sprawdź warunki rejestracji
            unidentified_faces = [res for res in session.face_results if res[0] == "Nieznany"]

            if len(unidentified_faces) != 1:
                self.show_message(
                    "Błąd rejestracji",
                    "Do rejestracji wymagana jest dokładnie jedna, nierozpoznana twarz w kadrze.",
                    Gtk.MessageType.WARNING
                )
                return

            # Rejestracja
            registered = self.system.face_recognition.register_face(frame, user_id)
            if registered:
                self.show_message("Sukces", f"Pomyślnie zarejestrowano użytkownika '{user_id}'.")
            else:
                self.show_message("Błąd", "Wystąpił nieoczekiwany błąd podczas rejestracji.", Gtk.MessageType.ERROR)
    
    def on_users_clicked(self, button):
        dialog = Gtk.Dialog(title="Zarejestrowani użytkownicy", transient_for=self, flags=0)
        dialog.add_buttons(Gtk.STOCK_CLOSE, Gtk.ResponseType.CLOSE)

        content_area = dialog.get_content_area()
        store = Gtk.ListStore(str)
        for user in sorted(self.system.face_recognition.get_registered_users()):
            store.append([user])

        treeview = Gtk.TreeView(model=store)
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn("Użytkownik", renderer, text=0)
        treeview.append_column(column)
        content_area.add(treeview)

        delete_button = Gtk.Button(label="Usuń zaznaczonego użytkownika")
        delete_button.connect("clicked", self.on_delete_user_clicked, treeview)
        content_area.add(delete_button)

        dialog.show_all()
        dialog.run()
        dialog.destroy()

    def on_delete_user_clicked(self, button, treeview):
        selection = treeview.get_selection()
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            user_id = model[treeiter][0]
            
            confirm_dialog = Gtk.MessageDialog(
                transient_for=self,
                flags=0,
                message_type=Gtk.MessageType.QUESTION,
                buttons=Gtk.ButtonsType.YES_NO,
                text=f"Czy na pewno chcesz usunąć użytkownika '{user_id}'?"
            )
            response = confirm_dialog.run()
            confirm_dialog.destroy()

            if response == Gtk.ResponseType.YES:
                if self.system.face_recognition.remove_user(user_id):
                    model.remove(treeiter)
                    self.show_message("Sukces", f"Usunięto użytkownika '{user_id}'.")
                else:
                    self.show_message("Błąd", f"Nie udało się usunąć użytkownika '{user_id}'.", Gtk.MessageType.ERROR)

    def show_message(self, title, text, msg_type=Gtk.MessageType.INFO):
        dialog = Gtk.MessageDialog(
            transient_for=self,
            flags=0,
            message_type=msg_type,
            buttons=Gtk.ButtonsType.OK,
            text=title
        )
        dialog.format_secondary_text(text)
        dialog.run()
        dialog.destroy()
    
    def on_destroy(self, *args):
        self.capture_thread.stop()
        
        self.system.cleanup()
        Gtk.main_quit()

def _detect_available_cameras(max_cameras_to_check=10) -> Dict[int, Any]:
    """Skanuje system w poszukiwaniu dostępnych kamer, używając backendu V4L2 dla lepszej kompatybilności z Linuksem."""
    available_cameras = {}
    logger.info(f"Rozpoczynanie detekcji kamer (sprawdzanie do {max_cameras_to_check} urządzeń przy użyciu backendu V4L2)...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w, _ = frame.shape
                available_cameras[i] = {"width": w, "height": h}
                logger.info(f"Znaleziono i zweryfikowano kamerę o ID {i} ({w}x{h})")
            else:
                logger.warning(f"Nie udało się odczytać klatki z kamery o ID {i}, mimo że jest 'otwarta'.")
            cap.release()
    logger.info(f"Zakończono detekcję. Znaleziono {len(available_cameras)} zweryfikowanych kamer.")
    return available_cameras

def _select_camera_gui(available: Dict[int, Any]) -> Optional[int]:
    """Wyświetla okno dialogowe GTK do wyboru kamery."""
    dialog = Gtk.Dialog(title="Wybierz kamerę", flags=0)
    dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OK, Gtk.ResponseType.OK)
    dialog.set_default_size(300, 150)

    box = dialog.get_content_area()
    label = Gtk.Label(label="Wybierz urządzenie wideo do użycia:")
    box.pack_start(label, True, True, 10)

    combo = Gtk.ComboBoxText()
    for cam_id, details in available.items():
        combo.append_text(f"Kamera {cam_id} ({details['width']}x{details['height']})")
    combo.set_active(0)
    box.pack_start(combo, True, True, 10)

    dialog.show_all()
    response = dialog.run()
    
    selected_id = None
    if response == Gtk.ResponseType.OK:
        active_index = combo.get_active()
        selected_id = list(available.keys())[active_index]

    dialog.destroy()
    return selected_id


def main():
    # 1. Wykryj kamery
    detected_cams = _detect_available_cameras()
    camera_id = None

    # 2. Logika wyboru kamery
    if not detected_cams:
        # Wyświetl błąd, jeśli nie znaleziono kamer
        dialog = Gtk.MessageDialog(
            transient_for=None, flags=0, message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK, text="Błąd krytyczny")
        dialog.format_secondary_text("Nie znaleziono żadnej działającej kamery. Aplikacja zostanie zamknięta.")
        dialog.run()
        dialog.destroy()
        logger.critical("Nie znaleziono żadnej działającej kamery.")
        sys.exit(1)
    elif len(detected_cams) > 1:
        # Wyświetl okno wyboru, jeśli jest więcej niż jedna kamera
        camera_id = _select_camera_gui(detected_cams)
        if camera_id is None:
            logger.info("Anulowano wybór kamery. Zamykanie aplikacji.")
            sys.exit(0)
    else:
        # Wybierz automatycznie, jeśli jest tylko jedna
        camera_id = list(detected_cams.keys())[0]
        logger.info(f"Automatycznie wybrano kamerę o ID: {camera_id}")

    # 3. Uruchom aplikację
    app = BiometricApp(camera_id=camera_id)
    app.show_all()
    Gtk.main()

if __name__ == "__main__":
    main()
