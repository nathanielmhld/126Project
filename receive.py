import threading
import sys
from array import array
from queue import Queue, Full
from collections import deque
from huffman_zlib import ZlibCoder
from huffman import HuffDict
from reedsolomon import RSCode
import matplotlib.pyplot as plt
import pyaudio
import time
import numpy as np
import scipy.signal as sig
import config

# load huffDict
huffDict = HuffDict.from_save(config.HUFF_DICT_PATH)
zlibCoder = ZlibCoder()
rsCode = RSCode()

SPLIT_CHUNKS = 2
TRANSMITTER_CHUNK_SIZE = config.CHUNK_SIZE
print('using chunk size:', TRANSMITTER_CHUNK_SIZE)

CHUNK_SIZE = TRANSMITTER_CHUNK_SIZE // SPLIT_CHUNKS
fs = config.SAMPLING_RATE
# if the recording thread can't consume fast enough, the listener will start discarding
BUF_MAX_SIZE = CHUNK_SIZE * 100

# if true, loads from file rather than actually listening
FROM_FILE = False
# if true, saves recording to file (only if FROM_FILE false and config.DEBUG true)
SAVE_TO_FILE = True

if len(sys.argv) > 1 and sys.argv[1] == '-f':
    FROM_FILE = True

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
        if SAVE_TO_FILE and config.DEBUG:
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
            print('* bitrate:', len(msg) * 8 / (exe_times[3] - exe_times[0]))
        print('bits in message:', len(msg) * 8)
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
                                          thresh = 0.01):
    ints, other_total = integrate_channels(freqs, mags, channel_freqs, freq_thresh)
    score = min(ints)
    if score == 0:
        if config.DEBUG:
            print('warning: check_has_freqs gave score of 0!')
        return None

    score = score / other_total
    if config.DEBUG:
        if channel_freqs[0] == config.START_SIGNAL[0][0]:
            print('awaiting start sig, intensities=', ints, ' noise=',  other_total, 'snr=', score)
        else:
            print('awaiting end sig, intensities=', ints, ' noise=',  other_total, 'snr=', score)
    if score < thresh:
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
    for chunk in process_q:
        all_data.extend(chunk)

    process_q.clear()
    while True:
        if stopped.wait(timeout=0):
            break
        while len(process_q) < SPLIT_CHUNKS:
            new_chunk = q.get()
            all_data.extend(new_chunk)
            process_q.append(new_chunk)
        #if config.DEBUG:
        #    print(len(all_data), 'samples received')
        chunk = to_float_audio(np.hstack(process_q))[0]

        process_q.popleft()
        data_fft, mags, freqs = fft(chunk)

        score = check_has_freqs(freqs, mags, config.END_SIGNAL[0], thresh = 0.01)
        if score:
            break
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

    STFT_STEP = config.STFT_STEP
    sf, st, sZ = sig.stft(audio, fs, nperseg=TRANSMITTER_CHUNK_SIZE, noverlap=TRANSMITTER_CHUNK_SIZE//STFT_STEP * (STFT_STEP-1))
    sZ_mag = np.absolute(sZ)
    sZ_phase = np.angle(sZ)
    exe_times.append(time.time())

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
    #print(channel_freqs_i)
    #print(all_channels)

    ch_start = np.searchsorted(sf, config.TRANSMITTER_STARTS[0] - config.TRANSMITTER_INTERVALS[0]/2)
    sZ_mag = sZ_mag[ch_start:, :]
    sf = sf[ch_start:]

    #ch_means =  np.mean(sZ_mag[channel_idxs, :], axis=1, keepdims=True)
    #ch_std =  np.std(sZ_mag[channel_idxs, :], axis=1, keepdims=True)
    #sZ_mag[channel_idxs, :] = (sZ_mag[channel_idxs, :] - ch_means) / ch_std + 1

    t_sum =  np.sum(sZ_mag, axis=0, keepdims=True)
    sZ_mag = sZ_mag / t_sum
    
    lo_sep = np.searchsorted(sf, 2090)
    hi_sep = np.searchsorted(sf, 2340)
    hi_cutoff = np.searchsorted(sf, 3900)
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
    im_to_show = sZ_mag[:].T
    ax.imshow(im_to_show, cmap='viridis')
    #ax.imshow(channel_mag.T, cmap='viridis')
    ax.set_yticks(np.arange(0, len(st), 300))
    ax.set_yticklabels(map(lambda x: '%.2f' % x, st[::300]))
    ax.set_xticks(np.arange(0, len(sf[:]), 70))
    ax.set_xticklabels(map(lambda x: '%.2f' % x, sf[::70]))
    #ax.set_xticks(np.arange(0, len(channel_idxs), 20))
    #ax.set_xticklabels(map(lambda x: '%.2f' % x, sf[channel_idxs[::20]]))
    #ax.set_aspect(0.1)
    ax.set_aspect(1)

    plt.figure()
    plt.title('max frequency')

    plt.plot(t_max_ch_lo)
    plt.plot(t_max_ch_hi)
    plt.show()
    """

    start_signal_end = 0
    while start_signal_end < len(t_max_ch_hi) and t_max_ch_hi[start_signal_end] < config.START_SIGNAL[0][0] - 100:
        start_signal_end += 1
    while start_signal_end < len(t_max_ch_hi) and t_max_ch_hi[start_signal_end] > config.START_SIGNAL[0][0] - config.TRANSMITTER_INTERVALS[1]:
        start_signal_end += 1
    if config.DEBUG:
        print('start signal ends:', start_signal_end)

    end_signal_start = len(t_max_ch_hi)-1
    while end_signal_start > 0 and t_max_ch_hi[end_signal_start] < config.END_SIGNAL[0][0] - 100:
        end_signal_start -= 1
    while end_signal_start > 0 and t_max_ch_hi[end_signal_start] > config.END_SIGNAL[0][0] - config.TRANSMITTER_INTERVALS[1]/2:
        end_signal_start -= 1
    if end_signal_start <= start_signal_end:
        end_signal_start = len(t_max_ch_hi)-1
    if config.DEBUG:
        print('end signal starts:', end_signal_start)

    sZ_mag_hi = sZ_mag[hi_sep:hi_cutoff, :]
    t_max_ch_i_hi =  np.argmax(sZ_mag_hi, axis=0)
    t_max_ch_hi = sf_hi[t_max_ch_i_hi]

    decoded, ignored = [], []
    last_exceed_range = False

    chunks_to_decode = (end_signal_start - start_signal_end - 1) // STFT_STEP + 1
    _prev_lo, _prev_hi = -1, -1

    # frequency to (half) byte
    decode_freq = lambda fr, trans: int(np.round((fr - config.TRANSMITTER_STARTS[trans]) / config.TRANSMITTER_INTERVALS[trans]))

    for i in range(chunks_to_decode):
        stft_offset = min(STFT_STEP // 4, 2)
        start_pos = start_signal_end + i*STFT_STEP + stft_offset
        end_pos = min(start_signal_end + (i+1)*STFT_STEP - stft_offset, end_signal_start+1)
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
            recon_hi = max(decode_freq(high_freq, 1), 0)
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
            recon_lo = max(decode_freq(low_freq, 0), 0)
            #print('!l ', past_low_freq, low_freq, _prev_lo, recon_lo)


        # detect byte continuation signal
        _recon_lo, _recon_hi = recon_lo, recon_hi
        if high_freq >= config.TRANSMITTER_STARTS[1] + (2**config.PACKET_BITS) * config.TRANSMITTER_INTERVALS[1] and high_freq < config.END_SIGNAL[0][0] - 100:
            recon_hi = _prev_hi
        if low_freq >= config.TRANSMITTER_STARTS[0] + (2**config.PACKET_BITS) * config.TRANSMITTER_INTERVALS[0] and low_freq < config.TRANSMITTER_STARTS[1]:
            recon_lo = _prev_lo
        if config.DEBUG:
            #print(t_max_ch_lo[start_pos : end_pos])
            #print(t_max_ch_hi[start_pos : end_pos])
            print("{0:.2f} {1:.2f}: {2} {3}".format(low_freq, high_freq, recon_lo, recon_hi))

        _prev_hi = _recon_hi
        _prev_lo = _recon_lo
        if recon_hi >= 16 or recon_lo >= 16:
            last_exceed_range = True
            if config.DEBUG:
                print('warning: channel out of acceptable range (0-{}), possibly beginning of end signal?'.format(2**config.PACKET_BITS-1))
                ignored.extend([recon_lo, recon_hi])
            continue

        if t_max_ch_hi[start_signal_end] > config.END_SIGNAL[0][0] - 100:
            last_exceed_range = True
            continue
        """
        elif best_eff_mag < 3.:
            print('ignored: signal too quiet')
            continue
        """
        last_exceed_range = False

        for recon in ignored:
            digit = (1 << (config.PACKET_BITS-1))
            i = config.PACKET_BITS-1
            while digit:
                decoded.append(((recon & digit) >> i) ^
                               ((config.MESSAGE_BITS-i) & 1))
                digit >>= 1
                i -= 1

        ignored = []

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
    #print(decoded)

    print('total bits received:', len(decoded))
    if config.DEBUG:
        import os.path
        if os.path.exists('_actual_message.npy'):
            actual = np.load('_actual_message.npy')
            while len(decoded) > len(actual):
                decoded.pop()
            while len(decoded) < len(actual):
                decoded.append(0)
            Y = np.array(decoded)
            X = np.array(actual)
            bitwise_errs = np.sum(np.abs(Y - X))
            X = X.reshape(-1, 8)
            Y = Y.reshape(-1, 8)
            errs = np.sum(np.any(X != Y, axis=1))
            #print(X)
            #print(Y)
            print('bit errors', bitwise_errs, '=', bitwise_errs / (len(decoded) * 8))
            print('byte errors', errs, '=', errs / len(decoded))

    rs_decode = rsCode.decode(decoded)
    #print(decoded)
    use_huff = False
    try:
        if rs_decode[-1] == 1:
            decompr = zlibCoder[rs_decode[:-1]]
            use_huff = False
        else:
            decompr = huffDict[rs_decode[:-1]]
            use_huff = True
    except:
        print('warning: compression scheme specified did not work, perhaps header is corrupted? Trying other scheme...')
        if rs_decode[-1] == 0:
            decompr = zlibCoder[rs_decode[:-1]]
            use_huff = False
        else:
            decompr = huffDict[rs_decode[:-1]]
            use_huff = True
    if use_huff:
        print('decompressed using custom huffman')
    else:
        print('decompressed using zlib')
    return decompr

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
