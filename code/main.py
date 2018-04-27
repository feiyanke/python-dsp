import env
import block


sin = block.SinSignal(amp=0.8, freq=440)
cos = block.CosSignal(amp=0.8, freq=440)
buffer1 = block.Buffer()
buffer2 = block.Buffer()
audio = block.Audio(1, env.sim_fs)
scope1 = block.Scope()
scope2 = block.Scope()


class TestBlock(block.Block):
    def run(self):
        s = sin()
        scope1(s)
        # c = cos()
        # scope2(c)
        # buffer1(s)
        # buffer2(env.get_ts())
        # audio(s + c)


env.simulate(TestBlock())
