# Import
# Jussi Kangas och Gabriel Ellebrink
# jkangas@kth.se och elleb@kth.se

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


def createSpecto(fs, sound, M, freqRange):
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

            zeroIndx=np.where(freqCol == 0)
            if(len(zeroIndx[0])>=1):
                for index in zeroIndx:
                    freqCol[index]=1

            freqCol = np.abs(np.log(freqCol**2))
            spectogram[:, j] = freqCol
        N += M
    sound_time = len(sound) / fs
    time_array = np.linspace(0, sound_time, spectogram.shape[1])

    # Vi hittar max frekvens för att få rätt skala

    maxFreq=max(freqRange)
    freq_array = np.linspace(0, maxFreq, spectogram.shape[0])

    return time_array, freq_array, spectogram

def spectoPlot(fs, sound, M, freqRange):
    """
    Startar hela skiten samt plottar.
    :param sound: Insignal i tidsdomänen.
    :return: None
    """
    spectogram = createSpecto(fs, sound, M, freqRange)

    plt.pcolormesh(spectogram[0], spectogram[1], spectogram[2])
    plt.show()

    """"
    #3D kurva
    SpectoLine = spectogram[2].ravel()
    time = np.linspace(0, max(spectogram[0]), len(SpectoLine))
    freq = np.linspace(0, max(spectogram[1]), len(SpectoLine))
    ax = plt.axes(projection='3d')
    ax.plot3D(time, freq, SpectoLine)
    plt.show()

    #3D yta
    X, Y = np.meshgrid(spectogram[0], spectogram[1])
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, spectogram[2], rstride=1, cstride=1,
                    cmap='plasma', edgecolor='none')
    plt.show()
    """

def main():
    plt.close()
    fs, sound = wavfile.read('cantina.wav')

    # Numpys egna spectrogram
    f, t, Sxx = signal.spectrogram(sound, fs)

    zeroIndx = np.where(Sxx == 0)
    y = zeroIndx[0]
    x = zeroIndx[1]
    for i in range(0, len(x)):
        if (Sxx[y[i]][x[i]] == 0):
            Sxx[y[i]][x[i]] = 1

    Sxx = np.log(Sxx)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    #Vårat spectrogram
    spectoPlot(fs, sound, 500, f)



    """"
    X, Y = np.meshgrid(t, f)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Sxx, rstride=1, cstride=1,
                    cmap='plasma', edgecolor='none')

    plt.show()
    """
    """"
    SpectoLine=Sxx.ravel()
    time=np.linspace(0,max(t),len(SpectoLine))
    freq = np.linspace(0, max(f), len(SpectoLine))
    ax = plt.axes(projection='3d')
    ax.plot3D(time, freq, SpectoLine)
    plt.show()
    """



    #Vi plottar en sinuston
    sinus = sinusTone(2000, 8)


    #Sinuston med numpys egna
    f, t, Sxx = signal.spectrogram(sinus, 1/100)
    plt.pcolormesh(t, f, np.log(Sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    #Sinus med våran
    spectoPlot(1 / 100, sinus, 129,f)

    #3D plot
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')


    #X,Y=np.meshgrid(t,f)
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, Sxx, rstride=1, cstride=1,
    #                cmap='plasma', edgecolor='none')
    #plt.show()



main()