import sounddevice as sd
import numpy as np
import soundfile as sf

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


amp = 0.8
freq = 440
ts = np.linspace(0, 10, 80000, dtype='float32')
s = amp * np.sin(2 * np.pi * freq * ts)
f = 0
def cb(outdata, frames, time, status):
    global f
    outdata[:,0] = s[f:f+frames]
    f+=frames

def test4():
    stream = sd.OutputStream(samplerate=8000, channels=1, callback=cb)
    stream.start()
    sd.sleep(10000)
    stream.stop()

test4()

