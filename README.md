# System Biometryczny z Rozpoznawaniem Twarzy i Emocji

Zaawansowany system biometryczny umożliwiający rozpoznawanie użytkowników na podstawie twarzy oraz analizę ich emocji w czasie rzeczywistym.

## Funkcjonalności

- **Rozpoznawanie twarzy** - identyfikacja zarejestrowanych użytkowników
- **Rejestracja nowych użytkowników** - proste dodawanie nowych osób do systemu
- **Analiza emocji** - wykrywanie podstawowych emocji (szczęście, smutek, złość, zaskoczenie, itp.)
- **Interfejs użytkownika** - intuicyjny interfejs z podglądem kamery i wynikami analizy
- **Zarządzanie użytkownikami** - możliwość przeglądania i usuwania zarejestrowanych użytkowników

## Wymagania systemowe

- Python 3.8
- Kamera internetowa
- System operacyjny: Windows/Linux/macOS

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/twoj-projekt/biometric-system.git
   cd biometric-system
   ```

2. Zainstaluj wymagane pakiety:
   ```bash
   pip install -r requirements.txt
   ```

   **Uwaga:** Wymagane są dodatkowe zależności systemowe, szczególnie dla biblioteki `dlib` używanej przez `face_recognition`. 
   Instrukcje instalacji dla różnych systemów:
   - [Ubuntu/Debian](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
   - [macOS](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)
   - [Windows](https://www.learnopencv.com/install-dlib-on-windows/)

## Uruchomienie

```bash
python main_new.py
```

### Opcje wiersza poleceń

- `--camera ID` - Określa ID kamery (domyślnie: -1, co oznacza wybór kamery w GUI)

## Jak korzystać

1. **Wybór kamery**:
   - Po uruchomieniu programu bez parametrów (`python main.py`), system wykryje dostępne kamery
   - Jeśli wykryto więcej niż jedną kamerę, wyświetli się okno wyboru kamery, w którym możesz:
     - Używać klawiszy `n`/`p` lub strzałek lewo/prawo do przełączania między kamerami
     - Nacisnąć `Spację` lub `Enter` aby wybrać aktywną kamerę
     - Nacisnąć `Esc` aby anulować i wyjść z programu
   - Jeśli wykryto tylko jedną kamerę, zostanie ona wybrana automatycznie bez wyświetlania okna wyboru
   - Jeśli nie wykryto żadnej kamery, program zakończy działanie z komunikatem błędu

2. **Rejestracja nowego użytkownika**:
   - Uruchom program
   - Naciśnij `r` aby rozpocząć rejestrację
   - Wprowadź identyfikator użytkownika
   - Ustaw się przed kamerą i naciśnij `s` aby zrobić zdjęcie

2. **Rozpoznawanie użytkownika**:
   - System automatycznie rozpoznaje zarejestrowanych użytkowników
   - W lewym górnym rogu wyświetlany jest aktualny status
   - Wykrywane emocje są wyświetlane w prawym górnym rogu

3. **Zarządzanie użytkownikami**:
   - Naciśnij `l` aby wyświetlić listę zarejestrowanych użytkowników
   - Użytkowników można usunąć z poziomu kodu (funkcja w trakcie implementacji)

## Struktura projektu

- `main_new.py` - Główny plik uruchomieniowy
- `biometric_system.py` - Główna klasa systemu biometrycznego
- `face_utils.py` - Funkcje do rozpoznawania i rejestracji twarzy
- `emotion_analyzer.py` - Analiza emocji na podstawie punktów charakterystycznych twarzy
- `config.py` - Konfiguracja aplikacji
- `data/` - Katalog na dane użytkowników (tworzony automatycznie)
- `models/` - Katalog na modele ML (pobierane automatycznie)

## Licencja

Ten projekt jest dostępny na licencji MIT. Szczegóły w pliku LICENSE.

## Autor

Pająk Piotr

## Dodatkowe informacje

- Biblioteka MediaPipe za świetne modele do śledzenia twarzy
- Biblioteka face_recognition za łatwe w użyciu API do rozpoznawania twarzy
