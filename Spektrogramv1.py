# Import

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D


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

            freqCol = np.abs(np.log(freqCol**2))
            spectogram[:, j] = freqCol
        N += M
    sound_time = len(sound) / fs
    time_array = np.linspace(0, sound_time, spectogram.shape[1])

    # Vi hittar max frekvens för att få rätt skala
    freqs = np.fft.fftfreq(len(spectogram))

    max=float('Inf')
    spectogram2=spectogram
    count=0
    while( max == float('Inf') ):
        indxMax=np.unravel_index(np.argmax(spectogram2, axis=None), spectogram.shape)
        max=spectogram2[indxMax]
        if(max == float('Inf')):
            spectogram2[indxMax]=0
            count +=1


    freq=freqs[indxMax[0]+1]


    hertz=abs(freq*fs)

    freq_array = np.linspace(0, hertz, spectogram.shape[0])

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
    sinus = sinusTone(2000, 8)
    spectoPlot(1/100, sinus, 129)

    #Samma sinuston med numpys egna
    f, t, Sxx = signal.spectrogram(sinus, 1/100)
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    #3D plot
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    X,Y=np.meshgrid(f,t)
    Z = np.meshgrid(Sxx,t)
    testX=np.ones((5,7))
    testY=np.zeros((5,7))
    testZ=np.ones((5,7))*np.random.randint(0,10)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    #Axes3D.plot_surface(ax,testX,testY,testZ)
    plt.show()



main()