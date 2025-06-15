//main

Osoba osoba = new Osoba("Jan Nowak", 25);
osoba.View();


public class Osoba
{
    public string Imie { get; set; }
    public int Wiek { get; set; }

    public Osoba(string imie, int wiek)
    {
        Imie = imie;
        Wiek = wiek;
    }
    public void View()
    {
        System.Console.WriteLine($"Imię: {Imie}, wiek: {Wiek}");
    }
}

