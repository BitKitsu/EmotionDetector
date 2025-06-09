import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def test_camera_matplotlib(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Błąd: Nie można otworzyć kamery o ID {camera_id}")
        return
    
    print(f"Kamera {camera_id} otwarta pomyślnie")
    print(f"Rozdzielczość: {int(cap.get(3))}x{int(cap.get(4))}")
    
    # Konfiguracja wyświetlania matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    plt.title(f'Kamera {camera_id} - Zamknij okno, aby zakończyć')
    plt.axis('off')
    
    try:
        while plt.fignum_exists(fig.number):
            ret, frame = cap.read()
            if not ret:
                print("Błąd: Nie można odczytać klatki")
                break
                
            # Konwersja BGR (OpenCV) do RGB (matplotlib)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_display.set_array(frame_rgb)
            plt.draw()
            plt.pause(0.01)
            
    except KeyboardInterrupt:
        print("\nZatrzymywanie...")
    finally:
        cap.release()
        plt.close()
        print("Zasoby zwolnione")

if __name__ == "__main__":
    test_camera_matplotlib()
