import sounddevice as sd
import numpy as np
import soundfile as sf
from matplotlib.animation import TimedAnimation

def test1():
    data, fs = sf.read('1.wav', dtype='float32')
    sd.play(data, fs)
    sd.wait()

def test2():
    data, fs = sf.read('1.wav', dtype='float32')
    stream = sd.Stream(samplerate=fs, channels=2)
    stream.start()
    stream.write(data)
    stream.stop()

def test3():
    amp = 0.8
    freq = 440
    ts = np.linspace(0, 10, 80000, dtype='float32')

    stream = sd.OutputStream(samplerate=8000, channels=1)
    s = amp * np.sin(2 * np.pi * freq * ts)
    stream.start()
    stream.write(s)
    stream.stop()




def test4():

    amp = 0.8
    freq = 440
    ts = np.linspace(0, 10, 80000, dtype='float32')
    s = amp * np.sin(2 * np.pi * freq * ts)
    f = 0

    def cb(outdata, frames, time, status):
        global f
        outdata[:, 0] = s[f:f + frames]
        f += frames

    stream = sd.OutputStream(samplerate=8000, channels=1, callback=cb)
    stream.start()
    sd.sleep(10000)
    stream.stop()

def test5():
    import matplotlib.pyplot as plt
    ts = np.linspace(0, 0.02, 320, False)
    s = 0.8*np.sin(2 * np.pi * 440 * ts)
    fig = plt.figure()
    ax = plt.subplot()
    # a, = plt.plot(ts,s)
    plt.plot(np.linspace(0, 1, 500), np.linspace(0, 1, 500))
    plt.show()

def test6():
    import numpy as np
    import matplotlib.pyplot as plt

    fr = 20   #fft frequence resolution
    ts = 1 / fr  # sample time
    n = 20 #sample number
    fs = n / ts  # sample frequence Hz
    print('sample frequence:%f' % fs)

    f = 4   #signal frequence Hz
    step = 1/fs

    t = np.linspace(0, ts, n)
    s = np.sinc(2*np.pi*f*t)

    # np.sinc()
    # sfft = np.fft.fft(s)

    # ss = np.fft.ifft(sfft)

    srfft = np.fft.rfft(s)

    ssr = np.fft.irfft(srfft)

    # ff = np.fft.fftfreq(n, step)
    ffr = np.fft.rfftfreq(n, step)
    plt.figure()
    plt.plot(t, s, t, ssr)
    plt.figure()
    plt.plot(ffr, srfft.real, 'o')
    plt.show()

def test7():
    import numpy as np
    import matplotlib.pyplot as plt
    freq = 1
    phase = 0
    bias = 1
    ts = np.linspace(0, 20, 200)
    cycles = freq * ts + phase / np.pi / 2
    frac, _ = np.modf(cycles)
    ys = np.abs(frac - 0.5) * 2 - 0.5 + bias
    plt.figure()
    plt.plot(ts, frac)
    plt.show()


def test8():
    import matplotlib.pyplot as plt
    from numpy.fft import fft, fftshift
    window = np.hamming(51)
    plt.plot(window)
    plt.title("Hamming window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()
    plt.figure()
    A = fft(window, 1000) / 25.5
    mag = np.abs(fftshift(A))
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(mag)
    response = np.clip(response, -100, 100)
    plt.plot(freq, response)
    plt.title("Frequency response of Hamming window")
    plt.ylabel("Magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles per sample]")
    plt.axis('tight')
    plt.show()

def test9():
    import numpy as np
    ys = np.sin(np.linspace(0, 5, 100))
    window = np.ones(11)/11
    np.convolve(ys, window)

test8()

