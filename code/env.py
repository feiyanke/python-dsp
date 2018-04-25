import time
import numpy as np
from block import Block

sim_chunk = 320  # the number of samples for one loop
sim_fs = 16000  # sample rate in Hz

sim_time = 0  # current time in second
sim_begin_flag = False
sim_end_flag = False


def is_begin():
    return sim_begin_flag


def is_end():
    return sim_end_flag


# the duration of one loop
def get_duration():
    return sim_chunk / sim_fs


# get current loop of samples of time
def get_ts():
    return np.linspace(sim_time, sim_time + get_duration(), sim_chunk, False)


def simulate(block, t=10.0):
    assert isinstance(block, Block)

    global sim_time, sim_begin_flag, sim_end_flag

    print("simulation start")
    start_time = time.time()
    sim_time = 0
    while sim_time < t:
        print("simulation time: %f" % sim_time)
        if sim_time == 0:
            sim_begin_flag = True
        block.run()
        sim_begin_flag = False
        sim_time += get_duration()
    sim_time = t
    sim_end_flag = True
    block.run()
    sim_end_flag = False
    print("simulation end, time: %f s" % (time.time() - start_time))

