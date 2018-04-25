import env
import block

sin = block.SinSingal()
cos = block.CosSingal()
buffer = block.Buffer()
audio = block.Audio(1, env.sim_fs)

class TestBlock(block.Block):
    def run(self):
        s = sin()
        c = cos()
        buffer([s, c])
        audio(s+c)

env.simulate(TestBlock())
print(buffer.get(2).shape)
