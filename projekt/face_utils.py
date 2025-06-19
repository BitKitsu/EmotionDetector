import os
import cv2
import numpy as np
import face_recognition
from scipy.spatial import distance
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum


class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
from pathlib import Path
import pickle
import logging
from config import DATA_DIR, FACE_RECOGNITION_SETTINGS

logger = logging.getLogger(__name__)

class FaceRecognition:
    def __init__(self, tolerance: float = None, model: str = None):
        """Inicjalizacja systemu rozpoznawania twarzy.
        
        Args:
            tolerance: Tolerancja dopasowania twarzy (im mniejsza, tym bardziej restrykcyjne)
            model: Model do rozpoznawania twarzy ('hog' lub 'cnn')
        """
        # Ustawienie progów tolerancji dla różnych metryk
        self.tolerances = {
            DistanceMetric.COSINE: 0.15,  # Domyślna tolerancja (odpowiada 0.85 pewności)
            DistanceMetric.EUCLIDEAN: 0.35  # Domyślna tolerancja (odpowiada 0.65 pewności)
        }
        
        # Używamy HOG jako domyślnego modelu, ponieważ jest szybszy i działa dobrze na CPU
        self.model = 'hog' if model is None else model
        self.metric = DistanceMetric.COSINE  # Domyślna metryka
        self.tolerance = self.tolerances[self.metric] # Ustaw próg dla domyślnej metryki
        
        # Domyślne ustawienia wykrywania twarzy
        self.num_jitters = 1  # Mniej zaburzeń, szybsze
        self.upsample = 0     # Bez upsamplingu dla wydajności (im większa tym dokładniejsze, ale wolniejsze)
        
        # Inicjalizacja listy znanych twarzy
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Wczytanie zapisanych twarzy jeśli istnieją
        self.encodings_file = DATA_DIR / 'face_encodings.pkl'
        
        # Utwórz katalog na dane jeśli nie istnieje
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Wczytaj znane twarze
        self._load_known_faces()
        
        logger.info(f"Zainicjalizowano system rozpoznawania twarzy. Zarejestrowane twarze: {len(self.known_face_names)}")
    
    def _load_known_faces(self) -> None:
        """Wczytuje zapisane kody twarzy z pliku."""
        if self.encodings_file.exists():
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                logger.info(f"Wczytano {len(self.known_face_names)} zapisanych twarzy")
            except Exception as e:
                logger.error(f"Błąd podczas wczytywania zapisanych twarzy: {e}")
    
    def _save_known_faces(self) -> None:
        """Zapisuje kody twarzy do pliku."""
        try:
            with open(self.encodings_file, 'wb') as f:
                pickle.dump({
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names
                }, f)
            logger.info(f"Zapisano {len(self.known_face_names)} twarzy do pliku")
        except Exception as e:
            logger.error(f"Błąd podczas zapisywania twarzy: {e}")
    
    def register_face(self, image: np.ndarray, name: str) -> bool:
        """Rejestruje nową twarz w systemie.
        
        Args:
            image: Obraz wejściowy w formacie BGR
            name: Identyfikator osoby na zdjęciu
            
        Returns:
            bool: True jeśli udało się zarejestrować twarz, False w przeciwnym wypadku
        """
        logger.info(f"Rozpoczęcie rejestracji użytkownika: {name}")
        try:
            # Optymalizacja: skalowanie klatki przed detekcją
            small_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            rgb_image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            logger.debug("Konwersja obrazu z BGR do RGB i skalowanie zakończone")

            # Wykrywanie położenia twarzy na mniejszym obrazie
            logger.debug("Wykrywanie twarzy...")
            face_locations_small = face_recognition.face_locations(
                rgb_image, 
                model='hog',
                number_of_times_to_upsample=self.upsample
            )

            if not face_locations_small:
                logger.warning("Nie wykryto twarzy na zdjęciu")
                return False

            # Przeskaluj lokalizacje z powrotem do oryginalnego rozmiaru
            face_locations = [
                (top*2, right*2, bottom*2, left*2) for (top, right, bottom, left) in face_locations_small
            ]
            # Użyjemy pełnowymiarowego obrazu do ekstrakcji cech dla lepszej jakości
            rgb_image_full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug(f"Znaleziono {len(face_locations)} twarzy na zdjęciu")
            
            if not face_locations:
                logger.warning("Nie wykryto twarzy na zdjęciu")
                return False
            
            # Wybieramy największą twarz na zdjęciu
            face_locations = sorted(face_locations, 
                                 key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]), 
                                 reverse=True)
            
            # Ograniczamy się do jednej twarzy
            face_locations = face_locations[:1]
            
            # Pobranie cech twarzy z dodatkowymi parametrami
            logger.debug("Ekstrakcja cech twarzy...")
            face_encodings = face_recognition.face_encodings(
                rgb_image_full, 
                known_face_locations=face_locations,
                num_jitters=self.num_jitters,
                model='small'  # Użyj szybszego modelu dla wideo
            )
            logger.debug(f"Wyodrębniono cechy dla {len(face_encodings)} twarzy")
            
            if not face_encodings:
                logger.warning("Nie udało się wyodrębnić cech twarzy")
                return False
            
            # Sprawdzenie czy taka twarz już istnieje
            for i, face_encoding in enumerate(face_encodings):
                if self.known_face_encodings:  # Tylko jeśli mamy już jakieś twarze w bazie
                    logger.debug(f"Porównywanie twarzy {i+1} z bazą {len(self.known_face_encodings)} znanych twarzy...")
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    
                    # Znajdź najlepsze dopasowanie
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    
                    logger.debug(f"Najlepsze dopasowanie: {self.known_face_names[best_match_index]} "
                                 f"z odległością {best_distance:.4f} (próg: {self.tolerance})")
                    
                    if best_distance <= self.tolerance:
                        logger.info(f"Twarz już zarejestrowana jako: {self.known_face_names[best_match_index]} "
                                   f"(odległość: {best_distance:.4f})")
                        return False
            
            # Dodanie nowej twarzy
            self.known_face_encodings.extend(face_encodings)
            self.known_face_names.extend([name] * len(face_encodings))
            logger.info(f"Dodano nową twarz do bazy (łącznie: {len(self.known_face_names)} twarzy)")
            
            # Zapisanie zaktualizowanej listy twarzy
            self._save_known_faces()
            logger.info(f"Pomyślnie zarejestrowano nową twarz: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas rejestracji twarzy: {e}", exc_info=True)
            return False
    
    def recognize_face(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """Rozpoznaje twarze na podanym obrazie.
        
        Args:
            frame: Obraz wejściowy w formacie BGR
            
        Returns:
            Lista krotek (nazwa, pewność, (top, right, bottom, left)) dla każdej znalezionej twarzy
        """
        if not self.known_face_encodings:
            # Jeśli nie ma zarejestrowanych twarzy, zwróć wykryte twarze jako "Nieznane"
            try:
                # Konwersja z BGR do RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Wykrywanie twarzy - używamy HOG (szybsze) do wykrywania
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
                
                if not face_locations:
                    return []
                    
                # Zwróć wykryte twarze jako "Nieznane"
                return [("Nieznany", 0.0, loc) for loc in face_locations]
                
            except Exception as e:
                logger.error(f"Błąd podczas wykrywania twarzy: {e}")
                return []
            
        try:
            # Konwersja z BGR do RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Wykrywanie twarzy - używamy HOG (szybsze) do wykrywania
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            
            if not face_locations:
                return []
                
            # Pobranie cech twarzy z dodatkowymi parametrami
            face_encodings = face_recognition.face_encodings(
                rgb_frame, 
                known_face_locations=face_locations,  # Przetwarzamy wszystkie twarze
                num_jitters=self.num_jitters,
                model='small'  # Użyj szybszego modelu dla wideo
            )
            
            results = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Domyślne wartości dla nierozpoznanej twarzy
                name = "Nieznany"
                confidence = 0.0
                
                try:
                    if self.metric == DistanceMetric.EUCLIDEAN:
                        # Obliczenie odległości euklidesowej (oryginalna metoda biblioteki)
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings,
                            face_encoding
                        )
                    else:  # Domyślnie COSINE
                        # Obliczenie odległości kosinusowej (nasza implementacja)
                        face_distances = [distance.cosine(known_encoding, face_encoding) for known_encoding in
                                          self.known_face_encodings]

                    
                    # Znalezienie najlepszego dopasowania
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    
                    # Obliczenie pewności (im mniejsza odległość, tym większa pewność)
                    confidence = 1.0 - best_distance
                    
                    # Jeśli odległość jest mniejsza niż próg tolerancji, uznajemy za dopasowanie
                    if best_distance <= self.tolerance:
                        name = self.known_face_names[best_match_index]
                        logger.debug(f"Rozpoznano użytkownika: {name} (pewność: {confidence:.2f})")
                    else:
                        name = "Nieznany"
                        confidence = 0.0
                        logger.debug(f"Nie rozpoznano użytkownika (najlepsze dopasowanie: {self.known_face_names[best_match_index]} z odległością {best_distance:.4f})")
                except Exception as e:
                    logger.warning(f"Błąd podczas porównywania twarzy: {e}")
                    continue
                
                # Dodanie marginesu do wykrytej twarzy
                margin = 10
                h, w = frame.shape[:2]
                top = max(0, top - margin)
                right = min(w, right + margin)
                bottom = min(h, bottom + margin)
                left = max(0, left - margin)
                
                results.append((name, confidence, (top, right, bottom, left)))
            
            return results
            
        except Exception as e:
            logger.error(f"Błąd podczas rozpoznawania twarzy: {e}", exc_info=True)
            return []
    
    def get_registered_users(self) -> List[str]:
        """Zwraca listę unikalnych, posortowanych nazw zarejestrowanych użytkowników."""
        return sorted(list(set(self.known_face_names)))

    def get_tolerance(self, metric: DistanceMetric) -> float:
        """Pobiera próg tolerancji dla danej metryki."""
        return self.tolerances.get(metric, 0.6)  # Zwróć 0.6 jako domyślną wartość

    def set_tolerance(self, metric: DistanceMetric, value: float):
        """Ustawia próg tolerancji dla danej metryki."""
        if metric in self.tolerances:
            self.tolerances[metric] = value
            logger.info(f"Zaktualizowano próg tolerancji dla {metric.name} na: {value}")
    
    def set_metric(self, metric: DistanceMetric):
        """Ustawia metrykę do porównywania twarzy."""
        self.metric = metric
        self.tolerance = self.tolerances[self.metric]
        logger.info(f"Zmieniono metrykę porównywania na: {self.metric.value}")
        logger.info(f"Ustawiono próg tolerancji na: {self.tolerance}")

    def set_tolerance(self, metric: DistanceMetric, value: float):
        """Ustawia próg tolerancji dla podanej metryki."""
        if metric in self.tolerances:
            self.tolerances[metric] = value
            # Jeśli zmieniamy próg dla aktywnej metryki, zaktualizuj go natychmiast
            if self.metric == metric:
                self.tolerance = value
                logger.info(f"Zaktualizowano próg tolerancji dla {metric.value} na: {self.tolerance}")
    
    def toggle_metric(self) -> DistanceMetric:
        """Przełącza metrykę (COSINE <-> EUCLIDEAN) i aktualizuje próg tolerancji.
        Zwraca nową, aktywną metrykę."""
        if self.metric == DistanceMetric.COSINE:
            self.metric = DistanceMetric.EUCLIDEAN
        else:
            self.metric = DistanceMetric.COSINE
        self.tolerance = self.tolerances[self.metric]
        logger.info(f"Przełączono metrykę na: {self.metric.value}, próg: {self.tolerance}")
        return self.metric

    def remove_user(self, name: str) -> bool:
        """Usuwa użytkownika z bazy danych.
        
        Args:
            name: Identyfikator użytkownika do usunięcia
            
        Returns:
            bool: True jeśli użytkownik został usunięty, False w przeciwnym wypadku
        """
        if name not in self.known_face_names:
            logger.warning(f"Użytkownik {name} nie istnieje w bazie")
            return False
            
        try:
            # Znajdź wszystkie indeksy użytkownika
            indices = [i for i, x in enumerate(self.known_face_names) if x == name]
            
            # Usuń elementy w odwrotnej kolejności, aby uniknąć problemów z indeksami
            for i in sorted(indices, reverse=True):
                del self.known_face_encodings[i]
                del self.known_face_names[i]
            
            # Zapisz zmiany
            self._save_known_faces()
            logger.info(f"Usunięto użytkownika: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Błąd podczas usuwania użytkownika {name}: {e}")
            return False
