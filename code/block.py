import env
import numpy as np
from ndqueue import ndqueue
import sounddevice as sd


class Block(object):

    def __init__(self):
        self._time = -1.0
        self._y = None

    def __call__(self, *args, **kwargs):
        if env.sim_time > self._time:
            if env.is_begin():
                self.begin()
            self._y = self.run(*args, **kwargs)
            self._time = env.sim_time
            if env.is_end():
                self.end()
        return self._y

    def begin(self):
        pass

    def end(self):
        pass

    def run(self, *args, **kwargs):
        raise Exception("not implement run")


class FunctionBlock(Block):
    def __init__(self, f):
        super().__init__()
        if hasattr(f, '__call__'):
            self._f = f
        else:
            raise Exception("not function")

    def run(self, *args, **kwargs):
        raise self._f(*args, **kwargs)


class TimedFunctionBlock(FunctionBlock):
    def run(self, *args, **kwargs):
        raise self._f(env.get_ts(), *args, **kwargs)

# def sin_wave(amp=1.0, freq=1.0, phase=0.0, bias=0.0):
#     return lambda t: amp*np.sin(2*np.pi*freq*t+phase)+bias
#
# def cos_wave(amp=1.0, freq=1.0, phase=0.0, bias=0.0):
#     return lambda t: amp*np.sin(2*np.pi*freq*t+phase)+bias
#
# def sin_signal(amp=1.0, freq=1.0, phase=0.0, bias=0.0):
#     return TimedFunctionBlock(sin_wave())


class SinSingal(Block):

    def __init__(self, amp=1.0, freq=1.0, phase=0.0, bias=0.0):
        super().__init__()
        self._amp = amp
        self._freq = freq
        self._phase = phase
        self._bias = bias

    def run(self):
        return self._amp*np.sin(2*np.pi*self._freq*env.get_ts()+self._phase)+self._bias


class CosSingal(Block):

    def __init__(self, amp=1.0, freq=1.0, phase=0.0, bias=0.0):
        super().__init__()
        self._amp = amp
        self._freq = freq
        self._phase = phase
        self._bias = bias

    def run(self):
        return self._amp*np.cos(2*np.pi*self._freq*env.get_ts()+self._phase)+self._bias


class Buffer(Block):

    def __init__(self):
        super().__init__()
        self._buffer = []

    def run(self, s):
        self._buffer.append(s)

    def get(self, channel):
        result = np.asarray(self._buffer)
        result.shape = int(result.size / channel), channel
        return result


class Audio(Block):

    def __init__(self, channel, samplerate, dtype='float32'):
        super().__init__()

        def callback(outdata, frames, time, status):
            assert not status
            data = self._queue.dequeue(frames)
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            else:
                outdata[:] = data

        self._queue = ndqueue(channel, dtype=dtype)
        self._stream = sd.OutputStream(channels=channel, samplerate=samplerate, dtype=dtype, callback=callback)

    def run(self, s):
        self._queue.enqueue(s)

    def begin(self):
        self._stream.start()

    def end(self):
        self._stream.stop()

