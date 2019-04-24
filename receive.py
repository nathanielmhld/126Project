import threading
from array import array
from queue import Queue, Full
from collections import deque
from huffman_zlib import HuffDict
from reedsolomon import RSCode
import matplotlib.pyplot as plt
import pyaudio
import time
import numpy as np
import scipy.signal as sig
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode(allow_partial_block = config.RS_ALLOW_PARTIAL_BLOCK)

SPLIT_CHUNKS = 3
TRANSMITTER_CHUNK_SIZE = int(config.PACKET_TIME * config.SAMPLING_RATE)
print('using chunk size:', TRANSMITTER_CHUNK_SIZE)

CHUNK_SIZE = TRANSMITTER_CHUNK_SIZE // SPLIT_CHUNKS
fs = config.SAMPLING_RATE
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 100

# if true, loads from file rather than actually listening
FROM_FILE = False
# if true, saves recording to file (only if FROM_FILE = True)
SAVE_TO_FILE = True

p = pyaudio.PyAudio()

def to_float_audio(audio16):
    ''' convert to 16 bit float and split into channels '''
    return (audio16.astype(np.float32, order='C') / 32768.0).reshape(-1, 2).T

def main():
    exe_times = []
    if not FROM_FILE:
        stopped = threading.Event()
        q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))
        all_data = array('h', [])
        process_q = deque()

        listen_t = threading.Thread(target=listen, args=(stopped, q, process_q))
        listen_t.start()
        wait_for_end_t = threading.Thread(target=wait_for_end, args=(stopped, q, process_q, all_data, exe_times))
        wait_for_end_t.start()

        wait_for_end_t.join()
        stopped.set()
        listen_t.join()
        all_data = to_float_audio(np.array(all_data))
        np.save('sample_data.npy', all_data)
    else:
        exe_times.append(time.time())
        all_data = np.load('sample_data.npy')
    msg = decode(all_data, exe_times)
    exe_times.append(time.time())
    if msg is not None:
        print('\ndecoded message:')
        print(msg)
        print()
        if not FROM_FILE:
            print('* bitrate:', len(msg) * 8 / (exe_times[2] - exe_times[0]))
        print('time:', exe_times[-1] - exe_times[0])
        print('recd:', exe_times[1] - exe_times[0])
        print('stft:', exe_times[2] - exe_times[1])
        print('decd:', exe_times[3] - exe_times[2])
    else:
        print('decoded failed')
 
def fft(chunk):
    data_fft = np.fft.rfft(chunk, norm="ortho")
    mags = np.abs(data_fft)
    freqs = fs * np.fft.rfftfreq(len(chunk))
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
                                          mag_thresh = 0.3, other_thresh = 10.0):
    ints, other_total = integrate_channels(freqs, mags, channel_freqs, freq_thresh)
    score = min(ints)
    other_total /= score
    #print("The highest frequency is {} Hz".format(freqs[np.argmax(mags)]))
    if config.DEBUG:
        print(ints, other_total)
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
        chunk = to_float_audio(np.hstack(process_q))[0]
        process_q.popleft()
        data_fft, mags, freqs = fft(chunk)

        score = check_has_freqs(freqs, mags, config.START_SIGNAL[0])
        if score:
            return True
    return False

def wait_for_end(stopped, q, process_q, all_data, exe_times):
    """ thread to record until end signal """
    print("waiting for synchronization signal from transmitter...")
    while not has_started(q, process_q):
        if stopped.wait(timeout=0):
            break
    print("synchronized with transmitter, receiving data...")
    exe_times.append(time.time())
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
        chunk = to_float_audio(np.hstack(process_q))[0]

        process_q.popleft()
        data_fft, mags, freqs = fft(chunk)

        score = check_has_freqs(freqs, mags, config.END_SIGNAL[0],
                mag_thresh=0.3, other_thresh=10.0)
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

def play_data_32(data):
    """ plays 32-bit audio data, for testing """
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    stream.write(np.array(data).tobytes())

def decode(all_data, exe_times):
    exe_times.append(time.time())
    # use first channel for now
    audio = all_data[0]

    x_axis = np.arange(0, audio.size, dtype=np.float32) / fs
    #plt.figure()
    #plt.plot(x_axis, audio)

    STFT_STEP = 3
    sf, st, sZ = sig.stft(audio, fs, nperseg=TRANSMITTER_CHUNK_SIZE, noverlap=TRANSMITTER_CHUNK_SIZE//STFT_STEP * (STFT_STEP-1))
    sZ_mag = np.absolute(sZ)
    sZ_phase = np.angle(sZ)
    exe_times.append(time.time())

    """
    all_channels_lo = np.arange(config.TRANSMITTER_STARTS[0], config.TRANSMITTER_STARTS[0] +
                                (2**config.PACKET_BITS) * config.TRANSMITTER_INTERVALS[0], config.TRANSMITTER_INTERVALS[0])
    all_channels_lo = np.hstack([all_channels_lo, config.TRANSMITTER_CONTINUER[0] ])
    all_channels_hi = np.arange(config.TRANSMITTER_STARTS[1], config.TRANSMITTER_STARTS[1] +
                                (2**config.PACKET_BITS) * config.TRANSMITTER_INTERVALS[1], config.TRANSMITTER_INTERVALS[1])
    all_channels_hi = np.hstack([all_channels_hi, config.TRANSMITTER_CONTINUER[1] ])
    all_channels = np.hstack([all_channels_lo, all_channels_hi])
    channel_idxs = np.searchsorted(sf, all_channels)
    for i, c in enumerate(channel_idxs):
        if np.abs(all_channels[i] - sf[c-1]) < np.abs(all_channels[i] - sf[c]):
            channel_idxs[i] = c-1

    channel_freqs_i = sf[channel_idxs]
    """
    #print(channel_freqs_i)
    #print(all_channels)

    ch_start = np.searchsorted(sf, config.TRANSMITTER_STARTS[0] - config.FREQ_INTERVAL/2)
    sZ_mag = sZ_mag[ch_start:, :]
    sf = sf[ch_start:]

    #ch_means =  np.mean(sZ_mag[channel_idxs, :], axis=1, keepdims=True)
    #ch_std =  np.std(sZ_mag[channel_idxs, :], axis=1, keepdims=True)
    #sZ_mag[channel_idxs, :] = (sZ_mag[channel_idxs, :] - ch_means) / ch_std + 1

    t_sum =  np.sum(sZ_mag, axis=0, keepdims=True)
    sZ_mag = sZ_mag / t_sum
    
    lo_sep = np.searchsorted(sf, 1950)
    hi_sep = np.searchsorted(sf, 2250)
    sZ_mag_lo = sZ_mag[:lo_sep, :]
    sZ_mag_hi = sZ_mag[hi_sep:, :]

    sf_lo = sf[:lo_sep]
    sf_hi = sf[hi_sep:]

    t_max_ch_i_lo =  np.argmax(sZ_mag_lo, axis=0)
    t_max_ch_lo =  sf_lo[t_max_ch_i_lo]
    t_max_ch_i_hi =  np.argmax(sZ_mag_hi, axis=0)
    t_max_ch_hi =  sf_hi[t_max_ch_i_hi]

    """
    fig = plt.figure()
    plt.title('waterfall')
    ax = fig.add_subplot(111)
    ax.imshow(sZ_mag.T, cmap='viridis')
    #ax.imshow(channel_mag.T, cmap='viridis')
    ax.set_yticks(np.arange(0, len(st), 100))
    ax.set_yticklabels(map(lambda x: '%.2f' % x, st[::100]))
    ax.set_xticks(np.arange(0, len(sf), 30))
    ax.set_xticklabels(map(lambda x: '%.2f' % x, sf[::30]))
    #ax.set_xticks(np.arange(0, len(channel_idxs), 20))
    #ax.set_xticklabels(map(lambda x: '%.2f' % x, sf[channel_idxs[::20]]))
    #ax.set_aspect(0.1)
    ax.set_aspect(2)

    plt.figure()

    plt.plot(t_max_ch_lo)
    plt.plot(t_max_ch_hi)
    """

    start_signal_end = 0
    while t_max_ch_hi[start_signal_end] < config.START_SIGNAL[0][0] - 100:
        start_signal_end += 1
    while t_max_ch_hi[start_signal_end] > config.START_SIGNAL[0][0] - config.TRANSMITTER_INTERVALS[0]/2:
        start_signal_end += 1
    print('start signal ends:', start_signal_end)

    decoded = []
    last_exceed_range = False

    chunks_to_decode = (len(t_max_ch_hi) - start_signal_end - 2) // STFT_STEP + 1
    _prev_lo, _prev_hi = -1, -1

    # frequency to (half) byte
    decode_freq = lambda fr, trans: int(np.round((fr - config.TRANSMITTER_STARTS[trans]) / config.TRANSMITTER_INTERVALS[trans]))

    for i in range(chunks_to_decode):
        start_pos, end_pos = start_signal_end + i*STFT_STEP + 1, start_signal_end + (i+1)*STFT_STEP - 1
        high_freq = np.median(t_max_ch_hi[start_pos : end_pos])
        low_freq = np.median(t_max_ch_lo[start_pos : end_pos])
        if np.isnan(high_freq) or np.isnan(low_freq): break

        # reconstruction
        recon_hi = decode_freq(high_freq, 1)
        recon_lo = decode_freq(low_freq, 0)

        # impossible, by design
        if recon_hi == _prev_hi:
            # manually block out signal and try again
            #past_high_freq = high_freq
            high_freq_left = np.searchsorted(sf_hi, high_freq - config.TRANSMITTER_INTERVALS[1]/2)
            high_freq_right = np.searchsorted(sf_hi, high_freq + config.TRANSMITTER_INTERVALS[1]/2)

            sZ_mag_part = sZ_mag_hi[:, start_pos : end_pos]
            sZ_mag_part[high_freq_left : high_freq_right, :] = 0.0
            max_ch = sf_hi[np.argmax(sZ_mag_part, axis=0)]
            high_freq = np.median(max_ch)
            recon_hi = decode_freq(high_freq, 1)
            #print('!h ', past_high_freq, high_freq, _prev_hi, recon_hi)

        if recon_lo == _prev_lo:
            # manually block out signal and try again
            #past_low_freq = low_freq
            low_freq_left = np.searchsorted(sf_lo, low_freq - config.TRANSMITTER_INTERVALS[0]/2)
            low_freq_right = np.searchsorted(sf_lo, low_freq + config.TRANSMITTER_INTERVALS[0]/2)

            sZ_mag_part = sZ_mag_lo[:, start_pos : end_pos]
            sZ_mag_part[low_freq_left : low_freq_right, :] = 0.0
            max_ch = sf_lo[np.argmax(sZ_mag_part, axis=0)]
            low_freq = np.median(max_ch)
            recon_lo = decode_freq(low_freq, 0)
            #print('!l ', past_low_freq, low_freq, _prev_lo, recon_lo)


        # detect byte continuation signal
        if high_freq >= config.TRANSMITTER_STARTS[1] + (2**config.PACKET_BITS) * config.TRANSMITTER_INTERVALS[1] and high_freq < config.END_SIGNAL[0][0] - 100:
            recon_hi = _prev_hi
        if low_freq >= config.TRANSMITTER_STARTS[0] + (2**config.PACKET_BITS) * config.TRANSMITTER_INTERVALS[0] and low_freq < config.TRANSMITTER_STARTS[1]:
            recon_lo = _prev_lo
        if config.DEBUG:
            print("{0:.2f} {1:.2f}: {2} {3}".format(low_freq, high_freq, recon_lo, recon_hi))

        if recon_hi < 0 or recon_hi >= 16 or recon_lo < 0 or recon_lo >= 16:
            last_exceed_range = True
            if config.DEBUG:
                print('ignored: channel out of acceptable range (0-{}), possibly beginning of end signal?'.format(2**config.PACKET_BITS-1))
            continue
        _prev_hi = recon_hi
        _prev_lo = recon_lo

        if t_max_ch_hi[start_signal_end] > config.END_SIGNAL[0][0] - 100:
            last_exceed_range = True
            continue
        """
        elif best_eff_mag < 3.:
            print('ignored: signal too quiet')
            continue
        """
        last_exceed_range = False

        digit = (1 << (config.PACKET_BITS-1))
        i = config.PACKET_BITS-1
        while digit:
            decoded.append(((recon_lo & digit) >> i) ^
                           ((config.MESSAGE_BITS-i) & 1))
            digit >>= 1
            i -= 1

        digit = (1 << (config.PACKET_BITS-1))
        i = config.PACKET_BITS-1
        while digit:
            decoded.append(((recon_hi & digit) >> i) ^
                           ((config.MESSAGE_BITS-i) & 1))
            digit >>= 1
            i -= 1
        #print(decoded[-config.MESSAGE_BITS:], end=' ')
        #print()
    #if not last_exceed_range:
    #    decoded = decoded[:-8]
    #plt.show()
    #print(decoded)

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
