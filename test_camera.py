import cv2

def test_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Błąd: Nie można otworzyć kamery o ID {camera_id}")
        return
    
    print(f"Kamera {camera_id} otwarta pomyślnie")
    print(f"Rozdzielczość: {int(cap.get(3))}x{int(cap.get(4))}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Błąd: Nie można odczytać klatki")
            break
            
        cv2.imshow(f'Kamera {camera_id} - Naciśnij q, aby wyjść', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
