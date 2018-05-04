import env
import block
import numpy as np
import matplotlib.pyplot as plt


sin = block.SinSignal(amp=0.8, freq=440)
cos = block.CosSignal(amp=0.8, freq=440)
tri = block.TriangleSignal(amp=0.8, freq=440)
buffer1 = block.Buffer()
buffer2 = block.Buffer()
audio = block.Audio(1, env.sim_fs)
scope1 = block.Scope()
scope2 = block.Scope()
plt.figure()

class TestBlock(block.Block):
    def run(self):
        t = env.get_ts()
        s = tri()
        f = np.fft.rfft(s)
        a = np.abs(f)
        p = np.angle(f)
        ff = np.fft.rfftfreq(t.size, 1/env.sim_fs)
        plt.clf()
        plt.plot(ff, a*2/env.sim_chunk)
        plt.show()
        # scope1(s)
        # c = cos()
        # scope2(c)
        # buffer1(s)
        # buffer2(env.get_ts())
        # audio(s + c)


env.simulate(TestBlock())
