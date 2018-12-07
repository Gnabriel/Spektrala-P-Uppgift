# Import

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def sinusTone(x):
    """
    Skapar en sinuston med vald hastighet.
   :param x: Bestämmer hastigheten på sinusen
   :return: en sinuston
   """
    vinkel = np.linspace(0, np.pi * 2, 1000)
    sinus = np.sin(x * vinkel)
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


def createSpecto(sound_fft, M):
    """
    Skapar ett spektogram.
    :param sound: Insignal (i frek.domän)
    :param M: För varje ny kolumn flyttar man fönstret ett fixt antal sampel M
    :return: Spektogram (array)
    """
    N = len(sound_fft)
    i = 0
    spectogram = np.array(())
    while True:
        if i + M >= N:
            freq = averageFreq(sound_fft[i:N])  # ska man ta mean?
            break
        else:
            freq = averageFreq(sound_fft[i:i + M])

        spectogram = np.append(spectogram, freq)
        # spectogram = np.rint(spectogram) #Avrundar alla floats till ints
        i += M

    return spectogram


def main():
    fs, sound = wavfile.read('cantina.wav')
    N = len(sound)
    M = 300

    window = hammingWindow(N)
    sound_fft = directFourier(sound, N) * window
    # sound_fft = np.fft.fftshift(sound_fft)

    spectogram = createSpecto(sound_fft, M)
    spectogram = np.abs(spectogram)
    spectogram = np.fft.ifft(spectogram)

    sinus = sinusTone(1)
    plt.plot(np.fft.fft(sinus))
    plt.show()

    plt.plot(spectogram)
    plt.show()


main()