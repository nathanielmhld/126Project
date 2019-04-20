import threading
from array import array
from queue import Queue, Full
from collections import deque
from huffman import HuffDict
from reedsolomon import RSCode
import matplotlib.pyplot as plt
import pyaudio
import numpy as np
import scipy.signal as sig
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode()

SPLIT_CHUNKS = 8
CHUNK_SIZE = int(config.MESSAGE_DURATION * config.SAMPLING_RATE) // SPLIT_CHUNKS
#config.MESSAGE_DURATION
fs = config.SAMPLING_RATE
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 100

def main():
    stopped = threading.Event()
    q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))
    process_q = deque()

    listen_t = threading.Thread(target=listen, args=(stopped, q, process_q))
    listen_t.start()
    record_t = threading.Thread(target=record, args=(stopped, q, process_q))
    record_t.start()

    try:
        while True:
            listen_t.join(0.1)
            record_t.join(0.1)
    except KeyboardInterrupt:
        stopped.set()

    listen_t.join()
    record_t.join()
    
def fft(chunk):
    data_fft = np.fft.rfft(chunk, norm="ortho")
    mags = np.abs(data_fft)
    freqs = fs * 2 * np.fft.rfftfreq(len(chunk))
    return data_fft, mags, freqs

def integrate_channels(freqs, mags, channel_freqs, freq_thresh = config.FREQ_THRESH):
    ints = np.zeros(len(channel_freqs))
    other_total, j = 0.0, 0
    for i, f in enumerate(freqs):
        if abs(f - channel_freqs[j]) < freq_thresh:
            ints[j] += mags[i] / freq_thresh/ len(freqs) * fs
        else:   
            other_total += mags[i] / freq_thresh / len(freqs) * fs
            if f > channel_freqs[j] and j < len(channel_freqs)-1:
                j += 1
    return ints, other_total

def check_has_freqs(freqs, mags, channel_freqs, freq_thresh = config.FREQ_THRESH,
                                          mag_thresh = 50000.0, other_thresh = 8000000):
    ints, other_total = integrate_channels(freqs, mags, channel_freqs, freq_thresh)
    #print("The highest frequency is {} Hz".format(freqs[np.argmax(mags)]))
    #print(ints, other_total)
    score = min(ints)
    if score < mag_thresh:
        return None
    if other_total > other_thresh:
        return None
    return score

def wait_for_start(q, process_q, best_score=None):
    new_chunk = q.get()
    process_q.append(new_chunk)
    if len(process_q) >= SPLIT_CHUNKS:
        chunk = np.hstack(process_q)
        process_q.popleft()
        data_fft, mags, freqs = fft(chunk)

        score = check_has_freqs(freqs, mags, config.START_SIGNAL[0])
        if score:
            return score
        elif best_score:
            #process_q.clear()
            while len(process_q) > 2:
                process_q.popleft()
            return True
    return False

def record(stopped, q, process_q):
    movingaverage = []
    print("waiting for synchronization signal from transmitter...")
    best_wait_score = None
    while best_wait_score != True:
        if stopped.wait(timeout=0):
            break
        best_wait_score = wait_for_start(q, process_q, best_wait_score)
    print("synchronized with transmitter 1, receiving data...")
    decoded = []
    end_cnt, last_exceed_range = 0, False
    while end_cnt < 2:
        if stopped.wait(timeout=0):
            break

        while len(process_q) < SPLIT_CHUNKS:
            process_q.append(q.get())
        chunk = np.hstack(process_q)
        process_q.clear()

        data_fft, mags, freqs = fft(chunk)
        if check_has_freqs(freqs, mags, config.END_SIGNAL[0], mag_thresh=30000, other_thresh=30000000):
            end_cnt += 1
        else:
            end_cnt = 0
        #plt.plot(freqs[:2000], mags[:2000])
        #plt.savefig('lala.png')
        best_eff_mag, best_freq = 0, 320.0
        for i, f in enumerate(freqs):
            if f < config.LOW_FREQ - config.FREQ_INTERVAL/2: continue
            if f > 4000.0: break
            eff_mag = mags[i]/f**1.0
            if eff_mag > best_eff_mag:
                best_eff_mag = eff_mag
                best_freq = f

        high_freq = best_freq#freqs[np.argmax(mags)] 
        recon = int(np.round((high_freq - config.LOW_FREQ) / config.FREQ_INTERVAL))
        print("The highest frequency is {} Hz, channel {}".format(
              high_freq, recon))
        if recon < 0 or recon > 255:
            last_exceed_range = True
            print('ignored: received byte out of acceptable range (0-256)')
            continue
        last_exceed_range = False

        digit = (1 << (config.MESSAGE_BITS-1))
        i = config.MESSAGE_BITS-1
        while digit:
            decoded.append(((recon & digit) >> i) ^
                           ((config.MESSAGE_BITS-i) & 1))
            digit >>= 1
            i -= 1
        print(decoded[-config.MESSAGE_BITS:], end=' ')
        print()
    if not last_exceed_range:
        decoded = decoded[:-8]
    print("transmission ended")

    print('bits received:', len(decoded))
    test_rs_decode = rsCode.decode(decoded)
    test_decode = huffDict[test_rs_decode]
    print('decoded message:', test_decode)
    stopped.set()
    import os
    os._exit(1)

def listen(stopped, q, process_q):
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
            q.put(array('h', stream.read(CHUNK_SIZE, exception_on_overflow=False)))
        except Full:
            pass  # discard


if __name__ == '__main__':
    main()
