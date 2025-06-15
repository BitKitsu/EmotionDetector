% Wczytanie obrazu biometrycznego
I = imread('fingerprint.png');

% Sprawdzenie, czy obraz jest RGB czy grayscale
if size(I, 3) == 3
    I_gray = rgb2gray(I); % Konwersja do skali szarości
else
    I_gray = I; % Jeśli już jest w skali szarości, nie konwertujemy
end

% Wyświetlenie obrazu i jego histogramu
figure, imshow(I_gray), title('Obraz w skali szarości');
figure, imhist(I_gray), title('Histogram obrazu');
