I_eq = histeq(I_gray); % Wyrównanie histogramu

figure, imshow(I_eq), title('Obraz po wyrównaniu histogramu');
figure, imhist(I_eq), title('Histogram po wyrównaniu');