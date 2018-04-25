import numpy as np
import threading


class ndqueue(object):

    def __init__(self, channel=1, capacity=1024, dtype=float):
        assert channel > 0
        assert capacity > 0
        self._channel = channel
        self._buffer = []
        self._enq_condition = threading.Condition()
        self._deq_condition = threading.Condition()
        self._enq_size = 0
        self._deq_size = 0
        self._capacity = capacity
        self._size = 0
        self._dtype = dtype

    def enqueue(self, value, timeout=None):

        self._enq_condition.acquire()
        v = np.asarray(value)
        v.shape = int(v.size / self._channel), self._channel
        num = v.shape[0]
        assert num < self._capacity

        while self._size > self._capacity:
            self._enq_condition.wait(timeout)

        self._deq_condition.acquire()
        self._buffer.append(v)

        self._size += num
        if self._deq_size < self._size:
            self._deq_condition.notify_all()

        self._deq_condition.release()
        self._enq_condition.release()

    def dequeue(self, size, timeout=None):
        assert size < self._capacity
        self._deq_condition.acquire()

        while self._size < size:
            self._deq_condition.wait(timeout)

        self._enq_condition.acquire()

        result = np.ndarray((0, self._channel), self._dtype)
        while result.shape[0] < size:
            remain_size = size - result.shape[0]
            if self._buffer[0].shape[0] <= remain_size:
                result = np.append(result, self._buffer[0], axis=0)
                del self._buffer[0]
            else:
                s = self._buffer[0].shape[0]
                result = np.append(result, self._buffer[0][0:remain_size, :], axis=0)
                self._buffer[0] = self._buffer[0][remain_size:s, :]

        self._size -= size
        if self._size < self._capacity:
            self._enq_condition.notify_all()

        self._enq_condition.release()
        self._deq_condition.release()

        return result
