# Import
import numpy as np

def directFourier(x, N):
    """
    Transformerar en signal från tidsdomän till frek.domän.
    :param x: Insignal (array) i tidsdomän
    :param N: Längden av utsignalen
    :return: Utsignal (array) i frek.domän
    """
    return fft(x, n=N)

def pixelIntensity(X):
    """
    Räknar ut pixelintensiteten kring en tidpunkt.
    :param X: Signalen i frek.domänen.
    :return: Pixelintensitet i en array
    """
    return np.abs(np.log10(X))

def hammingWindow(N):
    """
    Skapar fönsterfunktionen "Hamming-fönster".
    :param N: Längden av x(n)
    :return: Fönsterfunktion som array
    """
    w = np.zeros((N))
    for n in range(N):
        w[n] = 0.53836 - 0.46164 * np.cos((2 * np.pi * n) / (N - 1))
    return w

def averageFreq(X):
    """
    Räknar ut medelvärdet av alla frekvenser i en viss tidpunkt.
    :param X: Signal i frekdomän
    :return: Medelvärde (float)
    """
    return np.sum(X) / len(X)

def main():
    array = ...
    fönster = hammingWindow(N)
    for sample in ljudfil:
        x_fft = directFourtier(x)
        x_apa = x_fft * fönster
        array[i] = averageFreq(x_apa)