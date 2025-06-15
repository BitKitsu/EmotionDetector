package ExampleTask;

import java.util.Scanner;

/**
 * 1. Napisz program, w którym zadeklarujesz tablicę 9-elementową. Wartości tablicy mają być wczytane z klawiatury
 * (wprowadzone przez użytkownika). Wartości wyświetl na ekranie w jednym wierszu. Sprawdź czy wartość środkowego
 * elementy czy (czwarty element tablicy) jest większy, mniejszy czy równy średniej arytmetycznej dwóch brzegowych
 * wartości (pierwszego i ostatniego). Wyświetl odpowiedni komunikat.
 */

public class Task1 {

    public static void main(String[] args) {
        run();

    }

    public static void run(){
        int [] array = new int[9];
        //wczytywanie danych z klawiatury
        for (int i = 0; i < array.length; i++) {
            array[i]=InputInt();
        }

        //wystwietlenie
        for (int item: array ) {
            System.out.print(item + " ");
        }
        System.out.print("\nWartosc środkowego elementu: "+ array[4]);
        float average = (array[0]+array[8])/2;
        if (array[4]>average) System.out.print(", jest większy od sredniej, która wynosi "+ average);
        else if (array[4]<average) System.out.print(", jest mniejszy od sredniej, która wynosi "+ average);
        else if (array[4]>=average) System.out.print(", jest równy sredniej, która wynosi "+ average);
    }

    public static int InputInt(){
        Scanner scanner = new Scanner(System.in);
        System.out.println("Podaj liczbe: ");
        int number = scanner.nextInt();
        return number;
    }
}
