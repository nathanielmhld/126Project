import threading
from array import array
from queue import Queue, Full
from collections import deque
from huffman import HuffDict
from reedsolomon import RSCode
import matplotlib.pyplot as plt
import pyaudio
import pickle
import time
import numpy as np
import scipy.signal as sig
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode(allow_partial_block = config.RS_ALLOW_PARTIAL_BLOCK)

SPLIT_CHUNKS = 8
TRANSMITTER_CHUNK_SIZE = int(config.MESSAGE_DURATION * config.SAMPLING_RATE)
CHUNK_SIZE = TRANSMITTER_CHUNK_SIZE // SPLIT_CHUNKS
#config.MESSAGE_DURATION
fs = config.SAMPLING_RATE
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 100

# if true, loads from file rather than actually listening
FROM_FILE = False
# if true, saves recording to file (only if FROM_FILE = True)
SAVE_TO_FILE = True

p = pyaudio.PyAudio()

def main():
    if not FROM_FILE:
        stopped = threading.Event()
        q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))
        all_data = array('h', [])
        process_q = deque()

        listen_t = threading.Thread(target=listen, args=(stopped, q, process_q))
        listen_t.start()
        wait_for_end_t = threading.Thread(target=wait_for_end, args=(stopped, q, process_q, all_data))
        wait_for_end_t.start()

        wait_for_end_t.join()
        stopped.set()
        listen_t.join()
        pickle.dump(all_data, open('sample_data.pkl', 'wb'))
    else:
        all_data = pickle.load(open('sample_data.pkl', 'rb'))
    msg = decode(all_data)
    if msg is not None:
        print('decoded message:', msg)
    else:
        print('decoded failed')
    
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
                                          mag_thresh = 20000.0, other_thresh = 8000000):
    ints, other_total = integrate_channels(freqs, mags, channel_freqs, freq_thresh)
    #print("The highest frequency is {} Hz".format(freqs[np.argmax(mags)]))
    #print(ints, other_total)
    score = min(ints)
    if score < mag_thresh:
        return None
    if other_total > other_thresh:
        return None
    return score

def has_started(q, process_q):
    """ returns true if start signal received """
    new_chunk = q.get()
    process_q.append(new_chunk)
    if len(process_q) >= SPLIT_CHUNKS:
        chunk = np.hstack(process_q)
        process_q.popleft()
        data_fft, mags, freqs = fft(chunk)

        score = check_has_freqs(freqs, mags, config.START_SIGNAL[0])
        if score:
            return True
    return False

def wait_for_end(stopped, q, process_q, all_data):
    """ thread to record until end signal """
    print("waiting for synchronization signal from transmitter...")
    while not has_started(q, process_q):
        if stopped.wait(timeout=0):
            break
    print("synchronized with transmitter, receiving data...")
    end_cnt = 0
    for chunk in process_q:
        all_data.extend(chunk)

    process_q.clear()
    while end_cnt < 2:
        if stopped.wait(timeout=0):
            break
        while len(process_q) < SPLIT_CHUNKS:
            new_chunk = q.get()
            all_data.extend(new_chunk)
            process_q.append(new_chunk)
        print(len(all_data), 'samples received')
        chunk = np.hstack(process_q)
        process_q.popleft()
        data_fft, mags, freqs = fft(chunk)

        score = check_has_freqs(freqs, mags, config.END_SIGNAL[0],
                mag_thresh=30000, other_thresh=40000000)
        if score:
            end_cnt += 1
        else:
            end_cnt = 0
        process_q.clear()
    print("finished receiving data, decoding...")

def play_data(data):
    """ plays 16-bit audio data, for testing """
    stream = p.open(format=p.get_format_from_width(2),
                    channels=2,
                    rate=fs,
                    output=True)
    stream.write(np.array(data).tobytes())

def find_start_signal_end(all_data):
    q = deque()
    BLOCK_SIZE = CHUNK_SIZE * SPLIT_CHUNKS * 2 * len(config.START_SIGNAL)
    for i in range(BLOCK_SIZE-1):
        q.append(all_data[i])

    max_score = 0.
    end_count = 0
    for i in range(BLOCK_SIZE-1, len(all_data), CHUNK_SIZE):
        for j in range(CHUNK_SIZE):
            q.append(all_data[i])
        data_fft, mags, freqs = fft(np.array(q))
        high_freq = freqs[np.argmax(mags)] 
        #print("The highest frequency is {} Hz".format(high_freq))
        score = check_has_freqs(freqs, mags, config.START_SIGNAL[0],
                                mag_thresh=20000, other_thresh=150000000)
        if score:
            print(score)
            end_count += 1
            if end_count <= 2:
                continue
            if score > max_score:
                max_score = score
            elif max_score - score > (max_score**0.5) * 3.0:
                return i
        while len(q) > BLOCK_SIZE:
            q.popleft()
    return 0

def decode(all_data):
    start_signal_end = find_start_signal_end(all_data)
    print('start signal ends:', start_signal_end)
    decoded = []
    last_exceed_range = False

    chunks_to_decode = (len(all_data) - start_signal_end - 1) // (TRANSMITTER_CHUNK_SIZE * 2) + 1

    for i in range(chunks_to_decode):
        chunk = all_data[start_signal_end + i * TRANSMITTER_CHUNK_SIZE * 2 : start_signal_end + (i+1) * TRANSMITTER_CHUNK_SIZE * 2]
        chunk_trim = int(len(chunk)*0.05)
        chunk = chunk[chunk_trim: -chunk_trim]

        """
        play_data(chunk)
        time.sleep(0.05)
        play_data(chunk)
        time.sleep(0.05)
        play_data(chunk)
        time.sleep(0.1)
        """

        data_fft, mags, freqs = fft(chunk)
        #plt.plot(freqs[:2000], mags[:2000])
        #plt.savefig('lala.png')
        best_eff_mag, best_freq = 0, 320.0
        for i, f in enumerate(freqs):
            if f < config.LOW_FREQ - config.FREQ_INTERVAL/2: continue
            if f > 4000.0: break
            eff_mag = mags[i]/f**1.3
            if eff_mag > best_eff_mag:
                best_eff_mag = eff_mag
                best_freq = f

        high_freq = best_freq#freqs[np.argmax(mags)] 
        recon = int(np.round((high_freq - config.LOW_FREQ) / config.FREQ_INTERVAL))
        print("The highest frequency is {} Hz, channel used is {}".format(
              high_freq, recon))

        if recon < 0 or recon > 255:
            last_exceed_range = True
            print('ignored: channel out of acceptable range (0-255), possibly beginning of end signal?')
            continue
        """
        elif best_eff_mag < 3.:
            print('ignored: signal too quiet')
            continue
        """
        last_exceed_range = False

        digit = (1 << (config.MESSAGE_BITS-1))
        i = config.MESSAGE_BITS-1
        while digit:
            decoded.append(((recon & digit) >> i) ^
                           ((config.MESSAGE_BITS-i) & 1))
            digit >>= 1
            i -= 1
        #print(decoded[-config.MESSAGE_BITS:], end=' ')
        #print()
    if not last_exceed_range:
        decoded = decoded[:-8]

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
