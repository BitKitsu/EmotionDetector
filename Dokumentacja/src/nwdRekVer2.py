# ************************************************************
# Zoptymalizowany algorytm Euklidesa - wersja rekurencyjna 
# Dane wejściowe: a, b
# Dane wyjściowe: a 
# ************************************************************
def nwdrek(a,b):
    if b!=0:
        return nwdrek(b, a % b)
    return a