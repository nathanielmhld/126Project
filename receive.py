import threading
from array import array
from queue import Queue, Full
import matplotlib.pyplot as plt
import pyaudio
import numpy as np
import scipy.signal as sig



CHUNK_SIZE = 2400
MIN_VOLUME = 100
fs = 44000  
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 10


def main():
    stopped = threading.Event()
    q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))

    listen_t = threading.Thread(target=listen, args=(stopped, q))
    listen_t.start()
    record_t = threading.Thread(target=record, args=(stopped, q))
    record_t.start()

    try:
        while True:
            listen_t.join(0.1)
            record_t.join(0.1)
    except KeyboardInterrupt:
        stopped.set()

    listen_t.join()
    record_t.join()


def record(stopped, q):
    movingaverage = []
    while True:
        if stopped.wait(timeout=0):
            break
        chunk = q.get()
        vol = max(chunk)
        if vol >= MIN_VOLUME and len(chunk) > 10:
            # TODO: write to file
            data_fft = np.fft.rfft(chunk, norm="ortho")
            mags = np.abs(data_fft)
            freqs = np.fft.rfftfreq(len(chunk))
            print("The frequency is {} Hz".format(fs*2*freqs[np.argmax(mags)]))


def listen(stopped, q):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=2,
        rate=fs,
        input=True,
        frames_per_buffer=1024,
    )

    while True:
        if stopped.wait(timeout=0):
            break
        try:
            q.put(array('h', stream.read(CHUNK_SIZE)))
        except Full:
            pass  # discard


if __name__ == '__main__':
    main()