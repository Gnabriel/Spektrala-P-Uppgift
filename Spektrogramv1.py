# Import

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def sinusTone(x, tone_length):
    """
    Skapar en sinuston med vald hastighet.
   :param x: Bestämmer hastigheten på sinusen
   :return: en sinuston
   """
    vinkel = np.linspace(0,np.pi*2,100)
    sinus = np.sin(x*vinkel)
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

def createSpecto(fs, sound, M):
    """
    Skapar ett spektogram.
    :param sound: Insignal (i tidsdomän)
    :param M: För varje ny kolumn flyttar man fönstret ett fixt antal sampel M
    :return: Spektogram (array)
    """

    cols_amount = int(len(sound) / M)
    spectogram = np.zeros((M//2, cols_amount))
    N = 0
    for j in range(cols_amount):
        if N + M <= len(sound):
            soundCol = sound[N:N + M]* hammingWindow(M)
            freqCol = np.fft.fft(soundCol)[0:M//2]
            #freqCol = np.fft.fftshift(freqCol) * hammingWindow(M)
            freqCol = np.abs(np.log(freqCol**2))
            spectogram[:, j] = freqCol
        N += M
    sound_time = len(sound) / fs
    time_array = np.linspace(0, sound_time, spectogram.shape[1])
    freq_array = np.linspace(0, 80, spectogram.shape[0])

    #print(len(time_array))
    #print(len(freq_array))
    #print(spectogram.shape)
    return time_array, freq_array, spectogram

def spectoPlot(fs, sound, M):
    """
    Startar hela skiten samt plottar.
    :param sound: Insignal i tidsdomänen.
    :return: None
    """
    spectogram = createSpecto(fs, sound, M)

    plt.pcolormesh(spectogram[0], spectogram[1], spectogram[2])
    plt.show()


def main():
    fs, sound = wavfile.read('cantina.wav')

    #Vårat spectrogram
    spectoPlot(fs, sound, 129)

    # Numpys egna spectrogram
    f, t, Sxx = signal.spectrogram(sound, fs)
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    #Vi plottar en sinuston med våran egna
    sinus = sinusTone(1000, 8)
    spectoPlot(1/100, sinus, 129)

    #Samma sinuston med numpys egna
    f, t, Sxx = signal.spectrogram(sinus, 1/100)
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()



main()