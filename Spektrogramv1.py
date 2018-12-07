# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def sinustone(x):
    """

    :param x: Hastigheten på
    :return:
    """
    sinus=np.zeros(1000);
    vinkel=linspace()

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

def main():
    fs, sound = wavfile.read('cantina.wav')
    N = len(sound)
    M = 300

    window = hammingWindow(N)
    sound_fft = directFourier(sound, N) * window
    #sound_fft = np.fft.fftshift(sound_fft)

    i = 0
    spectogram = np.array(())
    while True:
        if i+M >= N:
            freq = averageFreq(sound_fft[i:N])  #ska man ta mean?
            break
        else:
            freq = averageFreq(sound_fft[i:i+M])

        spectogram = np.append(spectogram, freq)
        #spectogram = np.rint(spectogram) #Avrundar alla floats till ints
        i += M

    spectogram = np.fft.ifft(spectogram)

    print(len(spectogram))
    plt.plot(spectogram)
    plt.show()


main()