import env
import numpy as np
from ndqueue import ndqueue
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import Animation


class Block(object):

    def __init__(self):
        self._time = -1.0
        self._y = None

    def __call__(self, *args, **kwargs):
        if env.sim_time > self._time:
            if env.is_begin():
                self.begin(*args, **kwargs)
            self._y = self.run(*args, **kwargs)
            self._time = env.sim_time
            if env.is_end():
                self.end(*args, **kwargs)
        return self._y

    def begin(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
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


class SinSignal(Block):

    def __init__(self, amp=1.0, freq=1.0, phase=0.0, bias=0.0):
        super().__init__()
        self._amp = amp
        self._freq = freq
        self._phase = phase
        self._bias = bias

    def run(self):
        return self._amp * np.sin(2 * np.pi * self._freq * env.get_ts() + self._phase) + self._bias


class CosSignal(Block):

    def __init__(self, amp=1.0, freq=1.0, phase=0.0, bias=0.0):
        super().__init__()
        self._amp = amp
        self._freq = freq
        self._phase = phase
        self._bias = bias

    def run(self):
        return self._amp * np.cos(2 * np.pi * self._freq * env.get_ts() + self._phase) + self._bias

class TriangleSignal(Block):

    def __init__(self, amp=1.0, freq=1.0, phase=0.0, bias=0.0):
        super().__init__()
        self._amp = amp
        self._freq = freq
        self._phase = phase
        self._bias = bias

    def run(self):
        ts = env.get_ts()
        cycles = self._freq * ts + self._phase / np.pi / 2
        frac, _ = np.modf(cycles)

        ys = np.abs(frac - 0.5) * 4 - 1 + self._bias
        return ys


class Buffer(Block):

    def __init__(self):
        super().__init__()
        self._buffer = []

    def run(self, s):
        self._buffer.append(s)

    def get(self):
        result = np.asarray(self._buffer)
        result.shape = result.size
        return result


class Audio(Block):

    def __init__(self, channel, samplerate, dtype='float32'):
        super().__init__()

        def callback(outdata, frames, time, status):
            assert not status
            print("callback")
            data = self._queue.dequeue(frames)
            if len(data) < len(outdata):
                outdata[:len(data)] = data
                outdata[len(data):] = b'\x00' * (len(outdata) - len(data))
            else:
                outdata[:] = data

        self._queue = ndqueue(channel, dtype=dtype)
        self._stream = sd.OutputStream(channels=channel, samplerate=samplerate, dtype=dtype, callback=callback)

    def run(self, s):
        print("run")
        self._queue.enqueue(s)

    def begin(self, s):
        print("begin")
        self._stream.start()

    def end(self, s):
        print("end")
        self._stream.stop()


class ScopeAnimation(Animation):

    def __init__(self, scope, blit=True):
        super().__init__(scope._fig, scope, blit)
        self._drawn_artists = [scope._ln]

    def _draw_frame(self, framedata):
        # for a in self._drawn_artists:
        #     a.set_visible(True)
        pass

    def _step(self, *args):
        self._draw_next_frame(None, self._blit)
        return True

    def new_frame_seq(self):
        'Creates a new sequence of frame information.'
        # Default implementation is just an iterator over self._framedata
        return []

    def new_saved_frame_seq(self):
        'Creates a new sequence of saved/cached frame information.'
        # Default is the same as the regular frame sequence
        return []


class Scope(Block):

    def run(self, s):
        ts = env.get_ts()
        self._ax.set_xlim(env.sim_time, env.sim_time + env.get_duration())
        self._ax.set_ylim(-1, 1)
        self._ln.set_data(ts, s)
        if self._cb is not None:
            self._cb()
        plt.pause(0.001)

    def begin(self, *args, **kwargs):
        self._cb = None
        self._fig = plt.figure()
        self._ax = plt.subplot()
        self._ln, = plt.plot(np.linspace(0, 1, 500), np.linspace(0, 1, 500), animated=True)
        self._ani = ScopeAnimation(self)

    def add_callback(self, cb):
        self._cb = cb

    def remove_callback(self, cb):
        self._cb = None

    def start(self):
        pass

    def stop(self):
        pass
