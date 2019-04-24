import threading
from array import array
from queue import Queue, Full
from collections import deque
from huffman import HuffDict
from reedsolomon import RSCode
import matplotlib.pyplot as plt
import pyaudio
import time
import timeit
import numpy as np
import scipy.signal as sig
import bpsk
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode(allow_partial_block = config.RS_ALLOW_PARTIAL_BLOCK)

SPLIT_CHUNKS, PAD_CHUNKS = 10, 20
TRANSMITTER_CHUNK_SIZE = int(config.MESSAGE_DURATION * config.SAMPLING_RATE)
CHUNK_SIZE = TRANSMITTER_CHUNK_SIZE // SPLIT_CHUNKS
#config.MESSAGE_DURATION
fs = config.SAMPLING_RATE
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 100

# if true, loads from file rather than actually listening
FROM_FILE = True
# if true, saves recording to file (only if FROM_FILE = True)
SAVE_TO_FILE = True

p = pyaudio.PyAudio()
def main():
    time_start = 0
    if not FROM_FILE:
        stopped = threading.Event()
        q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))
        all_data = array('h', [])
        process_q = deque()

        listen_t = threading.Thread(target=listen, args=(stopped, q, process_q))
        listen_t.start()
        wait_for_end_t = threading.Thread(target=wait_for_end, args=(stopped, q, process_q, all_data, time_start))
        wait_for_end_t.start()

        wait_for_end_t.join()
        stopped.set()
        listen_t.join()
        all_data = np.array(all_data).astype(np.float32, order='C') / 32768.0
        if SAVE_TO_FILE:
            np.save('sample_data.npy', all_data)
    else:
        all_data = np.load('sample_data.npy')
    #play_data(all_data.reshape(-1,2).transpose()[0][10100:])
    msg = decode(all_data)
    end_time = time.time()
    if msg is not None:
        print('decoded message:', msg)
    else:
        print('decoded failed')
    if not FROM_FILE:
        dtime = timeit.default_timer() - time_start
        print('time', dtime)
        print('rate', len(msg) * 8 / dtime)
    
def check_has_freqs(chunk, channel_freqs, off_freqs = config.REF_FREQS,
                    snr_thresh = 15.0):
    mags, off_mags = [], []
    for f in channel_freqs:
        if len(chunk.shape) == 1:
            mags.append(np.absolute(bpsk.fft_freq(chunk, f)))
        else:
            for i in range(chunk.shape[0]):
                mags.append(np.absolute(bpsk.fft_freq(chunk[i], f)))
    for f in off_freqs:
        if len(chunk.shape) == 1:
            off_mags.append(np.absolute(bpsk.fft_freq(chunk, f)))
        else:
            for i in range(chunk.shape[0]):
                off_mags.append(np.absolute(bpsk.fft_freq(chunk[i], f)))
    snr = np.min(mags) / np.max(off_mags)
    return snr >= snr_thresh

def has_started(q, process_q):
    """ returns true if start signal received """
    new_chunk = q.get()
    process_q.append(new_chunk)
    if len(process_q) >= SPLIT_CHUNKS + PAD_CHUNKS:
        chunk = np.hstack(process_q).reshape(-1, 2).transpose()
        process_q.popleft()
        if check_has_freqs(chunk, config.START_SIGNAL[0], snr_thresh = 15.0):
            return True
    return False

def wait_for_end(stopped, q, process_q, all_data, time_start):
    """ thread to record until end signal """
    print("waiting for synchronization signal from transmitter...")
    while not has_started(q, process_q):
        if stopped.wait(timeout=0):
            break
    time_start = timeit.default_timer()
    print("synchronized with transmitter, receiving data...")
    end_cnt = 0
    for chunk in process_q:
        all_data.extend(chunk)

    process_q.clear()
    while end_cnt < 1:
        if stopped.wait(timeout=0):
            break
        while len(process_q) < SPLIT_CHUNKS:
            new_chunk = q.get()
            all_data.extend(new_chunk)
            process_q.append(new_chunk)
        print(len(all_data), 'samples received')
        chunk = np.hstack(process_q).reshape(-1, 2).transpose()
        process_q.popleft()

        if check_has_freqs(chunk, config.END_SIGNAL[0], snr_thresh = 10.):
            end_cnt += 1
        else:
            end_cnt = 0
        process_q.clear()
    print("finished receiving data, decoding...")

def play_data(data):
    """ plays 32-bit audio data, for testing """
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    stream.write(np.array(data).tobytes())

def play_data16(data):
    """ plays 16-bit audio data, for testing """
    stream = p.open(format=p.get_format_from_width(2),
                    channels=2,
                    rate=fs,
                    output=True)
    stream.write(np.array(data).tobytes())

def decode(all_data):
    all_data = np.array(all_data)
    all_data = all_data.reshape(-1, 2).transpose() # sep channels
    print(len(all_data[1])/fs)

    # use only left chnl for now
    audio = all_data[0]

    # get rid of DC/drift
    from scipy.signal import detrend
    audio = detrend(audio, bp = np.arange(0, audio.size, 100))

    import matplotlib.pyplot as plt

    chunk_size = TRANSMITTER_CHUNK_SIZE
    binsz = 150
    step = 12
    freq = config.CARRIER_FREQ
    pstart = 9000#len(audio) - 450 - binsz
    pend = 60500#len(audio) - 100 - binsz

    #vol = np.max(audio) * 0.8
    #play_data(np.repeat(audio[2000:10100], 1))
    #print('vol', vol)
    #audio /= vol
    sqr = np.square(audio)
    mag_sqr, ang_sqr = bpsk.stft_freq(sqr, freq*2, binsz, step, 0, audio.size)

    mag_thresh = np.mean(mag_sqr) * 0.5
    beacon_start, beacon_end = 0, audio.size
    mag_run, run_t, long_run = 0, False, -1
    for i in range(len(mag_sqr)+1):
        if i >= len(mag_sqr):
            mag = 0
        else:
            mag = mag_sqr[i]
        t = mag > mag_thresh
        if t != run_t:
            #print(run_t, mag_run, long_run, i)
            if run_t == 1:
                if long_run == -1:
                    long_run = mag_run
                    beacon_start = i - mag_run
                else:
                    long_run += mag_run
                beacon_end = i 
            else:
                if mag_run > long_run:
                    long_run = -1
                else:
                    long_run += mag_run
            mag_run = 0
            run_t = t
        else:
            mag_run += 1
    #print('bea',beacon_start, beacon_end)

    beacon_times = []
    for i, ang in enumerate(ang_sqr[beacon_start: beacon_end]):
        if mag_sqr[i + beacon_start] > mag_thresh:
            t = ((beacon_start + i) * step) / fs
            rang = -ang
            if ang > 0:
                rang = np.pi * 2 - ang
            b = t + rang / (2 * np.pi) / (freq*2) 
            if not beacon_times:
                beacon_times.append(b)
            else:
                if b - beacon_times[-1] > 1./freq * 0.25:
                    beacon_times.append(b)
                else:
                    beacon_times[-1] = (beacon_times[-1] + b) * 0.5


    beacon_times = np.array(beacon_times)

    beacon_first = np.fmod(beacon_times, 1. / freq * 0.5)
    beacon_first = beacon_first[beacon_first > np.mean(beacon_first) - np.std(beacon_first) * 0.5]
    toff = np.mean(beacon_first) * fs
    print(toff, np.std(beacon_first))
    print(np.max(beacon_first), np.min(beacon_first))

    #for i, b in enumerate(beacon_times[:-1]):
        #print(beacon_times[i+1] - b, 1./(beacon_times[i+1] - b))

    #mag, ang = bpsk.stft_freq(audio, freq, binsz, 1, pstart, pend+binsz-1)

    x_axis = np.linspace(pstart / fs, pend / fs, pend - pstart)
    x_all = np.linspace(0, audio.size / fs, audio.size)
    plt.figure()
    plt.plot(x_axis, audio[pstart:pend], label='sig')
    plt.plot(x_axis, sqr[pstart:pend], label='sqr')
    #plt.plot(x_axis[::100], ang_sqr[pstart//100:pend//100] / (2*np.pi), label='sqr_conv_ang')

    #plt.plot(np.linspace(0, audio.size / fs, mag_mask.size), mag_mask, label='mag_mask')
    #plt.plot(np.linspace(0, audio.size / fs, mag_mask.size), mag_mask, label='mag_mask')
    #plt.plot(np.linspace(0, audio.size / fs, mag_mask.size), mag_mask, label='mag_mask')

    beacon_show = beacon_times[np.logical_and(beacon_times >= pstart/fs, beacon_times < pend/fs)]
    #print(beacon_show)

    samps = np.zeros(pend-pstart)
    for i in range(len(beacon_times) - 1):
        if beacon_times[i] < pstart/fs:
            continue
        if beacon_times[i] >= pend/fs:
            break

        idx = int(np.round((beacon_times[i] - beacon_times[0]) * freq * 2))
        #print(idx)

        #off = np.mean(np.fmod(beacon_times[max(i-1000, 0):i+1], 1. / freq * 0.5)) * fs
        cur, nex = int(np.round(beacon_times[i] * fs - pstart)), int(np.round(beacon_times[i+1] * fs - pstart))
        if cur < 0:
            cur = 0
        if nex > samps.size:
            nex = samps.size
        #print(cur, nex)
        offset = np.fmod(toff * 0.5, fs / freq)

        """
        if samps[cur-1] >= 0:
            idx = 1
        else:
            idx = 0
        """
        base_samps = np.fmod(np.arange(cur, nex), fs / freq)
        samps[cur: nex] = 0.4 * np.cos(2*np.pi*(base_samps + offset)* freq / fs)# + np.pi * (1 - idx % 2))
        #base_samps = np.fmod(np.arange(0, nex-cur), fs / freq)
        #samps[cur: nex] = 0.4 * np.cos(2*np.pi*(base_samps)* freq / fs + np.pi * (1 - idx % 2))
    print(samps)
    plt.plot(x_axis, samps, label='align')
    plt.plot(x_axis, audio[pstart:pend]*samps, label='mul')

    #plt.scatter(beacon_show, np.ones(len(beacon_show)), c='r', s=1, label='beacons')
    for i in range(10):
        plt.scatter(beacon_show, (-1 + i / 5.0) * np.ones(len(beacon_show)), c='r', s=1)
    msgdur = config.MESSAGE_DURATION
    rs = np.floor((pstart - beacon_start*step) / fs / msgdur) * msgdur + beacon_start * step / fs
    re = np.floor((pend - beacon_start*step) / fs / msgdur) * msgdur + beacon_start * step / fs
    markers = np.arange(rs, re, msgdur)
    plt.scatter(markers, np.ones(markers.size), c='g', s=1, label='message_marker')
    plt.legend()
    plt.show()
    import sys
    sys.exit(0)


    decoded = bpsk.decode(all_data[0], config.CARRIER_FREQ, begin=10111)

    print('total bits in message:', len(decoded))
    test_rs_decode = rsCode.decode(decoded)
    test_decode = huffDict[test_rs_decode]
    return test_decode

def listen(stopped, q, process_q):
    stream = p.open(
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
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            q.put(array('h', data))
        except Full:
            pass  # discard


if __name__ == '__main__':
    main()
