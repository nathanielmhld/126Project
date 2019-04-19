import threading
from array import array
from queue import Queue, Full
import matplotlib.pyplot as plt
import pyaudio
import numpy as np
import scipy.signal as sig
import config


CHUNK_SIZE = int(config.MESSAGE_DURATION * config.SAMPLING_RATE)
MIN_VOLUME = 1600
fs = config.SAMPLING_RATE
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


plt.ion()
def record(stopped, q):
    movingaverage = []
    freq_channels = np.arange(config.LOW_FREQ, config.LOW_FREQ + config.FREQ_INTERVAL * config.FREQ_CHANNELS, config.FREQ_INTERVAL)
    decoded = []
    while True:
        if stopped.wait(timeout=0):
            break
        chunk = q.get()
        print(chunk)
        vol = max(chunk)
        if vol >= MIN_VOLUME and len(chunk) > 10:
            # TODO: write to file
            data_fft = np.fft.rfft(chunk, norm="ortho")
            mags = np.abs(data_fft)
            freqs = fs * 2 * np.fft.rfftfreq(len(chunk))
            #plt.plot(freqs)
            ints = [0.] * config.FREQ_CHANNELS
            #print(freqs)
            j = 0
            for i, f in enumerate(freqs):
                if abs(f - freq_channels[j]) < config.FREQ_THRESH:
                    ints[j] += mags[i] / config.FREQ_THRESH
                elif f > freq_channels[j] and j < config.FREQ_CHANNELS-1:
                    j += 1
            for i in range(config.FREQ_CHANNELS):
                if ints[i] > config.AMP_THRESH:
                    decoded.append(1)
                else:
                    decoded.append(0)
            print(decoded[-config.FREQ_CHANNELS:], end=' ')
            print()
            #print(fs*2 *freqs[mags.argsort()[-10:][::-1]].astype(np.float32) )
            #print("The highest frequency is {} Hz".format(freqs[np.argmax(mags)]))


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
