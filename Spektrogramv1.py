# Import

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def sinusTone(x, tone_length):
    """
    Skapar en sinuston med vald hastighet.
   :param x: Bestämmer hastigheten på sinusen
   :return: en sinuston
   """
    vinkel=np.linspace(0,np.pi*2,100)
    sinus=np.sin(x*vinkel)
    for i in range(tone_length):
        sinus = np.append(sinus, sinus)
    return sinus

def directFourier(x, N):
    """
    Transformerar en signal från tidsdomän till frek.domän.
    :param x: Insignal (array) i tidsdomän
    :param N: Längden av utsignalen
    :return: Utsignal (array) i frek.domän
    """
    return np.fft.fft(x, n=N)

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
    return np.mean(X)

def createSpecto(sound, M):
    """
    Skapar ett spektogram.
    :param sound: Insignal (i frek.domän)
    :param M: För varje ny kolumn flyttar man fönstret ett fixt antal sampel M
    :return: Spektogram (array)
    """
    #sound_fft = np.fft.fftshift(sound_fft)

    spectogram = np.zeros((M, int(len(sound) / M)))
    N = 0
    for j in range(int(len(sound) / M)):
        if N + M <= N:
            soundCol = sound[N:N + M]
            freqCol = np.fft.fft(soundCol) * hammingWindow(M)
            spectogram[:, j] = freqCol
        N += M

    return spectogram

def spectoPlot(sound):
    """
    Startar hela skiten samt plottar.
    :param sound: Insignal i tidsdomänen.
    :return: None
    """
    N = len(sound)
    M = 300

    window = hammingWindow(N)
    sound_fft = directFourier(sound, N) * window
    # sound_fft = np.fft.fftshift(sound_fft)

    spectogram = createSpecto(sound_fft, M)

    plt.imshow(spectogram)
    #plt.plot(spectogram)
    plt.show()


def main():
    fs, sound = wavfile.read('cantina.wav')
    spectoPlot(sound)

    sinus = sinusTone(1, 7)
    spectoPlot(sinus)



main()