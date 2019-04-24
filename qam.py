''' QAM: Quadrature amplitude modulation tools '''
import numpy as np
from scipy import signal
import config

start_msg_len = 3
chunk_size = int(config.MESSAGE_DURATION * config.SAMPLING_RATE)
fs = config.SAMPLING_RATE   # sampling rate, Hz, must be integer

def to_float_audio(samps):
    return np.frombuffer(samps, dtype=np.float32)

def fft_freq(audio, freq, tstart=0, tend=-1):
    ''' single frequency fourier transform (convolve with in-phase, 90deg offset sine waves) '''
    if tend == -1:
        tend = audio.size
    N = tend - tstart
    times = np.arange(N) / fs
    return np.sum(audio[tstart:tend] * (np.cos(freq*2*np.pi*times, dtype=np.complex)-\
            1j*np.sin(2*np.pi* freq *times, dtype=np.complex)))*2/N

def stft_freq(audio, freq, binsz, interval):
    ''' single frequency short-time fourier transform '''
    lrange = range(0, audio.size-binsz+1, interval)
    ret = np.zeros(len(lrange), dtype=np.complex)
    for t, i in enumerate(lrange):
        ret[t] = fft_freq(audio, freq, i, i + binsz)# np.sum(conv[i:i+binsz])
    return np.absolute(ret), np.angle(ret)

def bits2iq(bitstr, start=0, nbits = config.MESSAGE_BITS):
    ''' convert binary to (I,Q) '''
    i, q = 0, 0
    bsize_i = 2**(nbits // 2 - 1)
    bsize_q = bsize_i
    max_bsize = bsize_i
    for j in range(start, start+nbits, 2):
        if bitstr[j] == 1:
            i -= bsize_i
        else:
            i += bsize_i
            if j == start:
                bsize_i = -bsize_i
        if bitstr[j+1] == 1:
            q -= bsize_q
        else:
            q += bsize_q
            if j == start:
                bsize_q = -bsize_q
        bsize_q >>= 1
        bsize_i >>= 1
    return i / (max_bsize * 2), q / (max_bsize * 2)

def iq2bits(inphase, quad, nbits = config.MESSAGE_BITS):
    ''' convert (I,Q) encoding to binary '''
    bsize_i = 2**(nbits // 2 - 1)
    bsize_q = bsize_i
    i, q = int(inphase * bsize_i) * 2, int(quad * bsize_i) * 2
    i += 2 * int(inphase > 0) - 1
    q += 2 * int(quad > 0) - 1
    mi, mq, xori, xorq = 0, 0, 0, 0

    bitstr = []
    for j in range(0, nbits, 2):
        if i > mi:
            mi += bsize_i
            bitstr.append(0 ^ xori)
        else:
            mi -= bsize_i
            bitstr.append(1 ^ xori)
        if q > mq:
            mq += bsize_q
            bitstr.append(0 ^ xorq)
        else:
            mq -= bsize_q
            bitstr.append(1 ^ xorq)

        if j == 0:
            if i > 0:
                xori = 1
            if q > 0:
                xorq = 1
        bsize_q >>= 1
        bsize_i >>= 1

    return bitstr

_window = np.hstack([np.arange(0, chunk_size // 2, dtype=np.float32), \
                       np.arange(chunk_size - chunk_size // 2, 0, -1, dtype=np.float32)])
_window /= np.max(_window)
_window = np.sqrt(_window)

def samples_iq(inphase, quad, t=0, freq = config.CARRIER_FREQ):
    """ generate samples with given in-phase, quadrature, and carrier freq, note conversion to float32 array """
    #inphase = 0.25
    #quad = 0.25
    base_samps = np.arange(chunk_size, dtype=np.float32)
    t = np.fmod(t, fs / (chunk_size * freq))
    samps = np.sin(2*np.pi*(base_samps + t * chunk_size)* freq / fs) * inphase - np.cos(2*np.pi*(base_samps + t * chunk_size)* freq / fs) * quad
    samps *= config.VOLUME
    samps *= _window

    return samps.tobytes()

def encode(bits, i):
    """ encode the next len(CHANNEL_FREQS) bits using QAM """
    all_freqs = []
    inphase, quad = bits2iq(bits, i*config.MESSAGE_BITS, config.MESSAGE_BITS)
    #print('playing: ', inphase, quad, i, bits[i*config.MESSAGE_BITS:(i+1)*config.MESSAGE_BITS])
    return samples_iq(inphase, quad, t=i+start_msg_len)

def recover_iq(audio, freq, begin = 0, ends = -1, iq_chunk_size = chunk_size):
    ''' recover the in-phase and quadrature components carried in signal '''
    if ends == -1:
        ends = audio.size
    N = ends - begin
    if N % iq_chunk_size:
        ends = audio.size - N % iq_chunk_size
        N = ends - begin

    times = np.arange(N) / fs
    conv = audio[begin:ends] * (np.cos(freq*2*np.pi*times, dtype=np.complex)-\
            1j*np.sin(2*np.pi* freq *times, dtype=np.complex))

    nyq = fs * 0.5
    b, a = signal.butter(3, freq * 1.5 / nyq, btype='low')
    re = -signal.lfilter(b, a, np.real(conv)) * 2
    im = -signal.lfilter(b, a, np.imag(conv)) * 2
    if re.size % iq_chunk_size:
        re = re[re.size % iq_chunk_size:]
        im = im[im.size % iq_chunk_size:]
    re = re.reshape(-1, iq_chunk_size)
    im = im.reshape(-1, iq_chunk_size)
    return np.mean(im, axis=1), np.mean(re, axis=1)

_bsize = 2**(config.MESSAGE_BITS // 2)
_max_amp = (_bsize-1) / _bsize

def decode(audio, freq, begin = -1):
    ''' recover binary signal from iq, use begin = -1 to auto find start signal '''
    if begin < 0:
        begin = find_start(audio, freq)
    I, Q = recover_iq(audio, freq, begin)

    # account for amplitude damping
    vol = np.mean(np.abs(I[:start_msg_len])) / _max_amp
    #print(vol)
    I /= vol
    Q /= vol
    #print(I, Q)

    decoded = []
    for i, q in zip(I[start_msg_len:], Q[start_msg_len:]):
        decoded.extend(iq2bits(i, q))
    return decoded

sig_iq = []
_start_sig, _start_sig_iq = None, None

def find_start(audio, freq):
    ''' find position of start signal '''
    # find noise level
    off_mags = []
    for f in config.REF_FREQS:
        off_mags.append(np.absolute(fft_freq(audio, f)))
    noise = np.mean(off_mags) * fs
    print('n',noise)

    coarse_pos = -1
    while coarse_pos < len(audio):
        # find coarse position
        stft = stft_freq(audio, freq, chunk_size // 9, chunk_size // 9)
        #_stft.append(stft)

        binsz = chunk_size // 9

        COARSE_MAG_THRESH = 0.0025
        FINE_Q_THRESH = 5e-3
        FINE_I_THRESH = 1e-2
        FINE_MAX_POS = 1000

        for i in range(coarse_pos + 1, audio.size-binsz+1, chunk_size // 9):
            conv = fft_freq(audio, freq, i, i + binsz)
            #mag = np.abs(conv) / noise
            mag = np.abs(conv) / noise
            print(i, mag)
            if mag > COARSE_MAG_THRESH:
                coarse_pos = i
                break            

        print('coarse', coarse_pos)

        # fine-tune position
        pos = -1
        for i in range(FINE_MAX_POS):
            I, Q = recover_iq(audio, freq, coarse_pos + i, coarse_pos + i + chunk_size * 3, iq_chunk_size = chunk_size)
            #if coarse_pos + i >= 5500 and coarse_pos + i <= 6900:
            print(coarse_pos + i, Q, I, np.min(np.abs(I)), np.max(np.abs(Q)))
            if np.min(np.abs(Q)) < FINE_Q_THRESH / noise and \
                 I[0] > FINE_I_THRESH / noise and \
                 I[1] < -FINE_I_THRESH / noise and \
                 I[2] > FINE_I_THRESH / noise:
                pos = coarse_pos + i
                break
        if pos >= 0:
            print('fine', pos)
            break
        else:
            coarse_pos += FINE_MAX_POS
    return pos

def start_sig(freq = config.CARRIER_FREQ):
    global _start_sig, _start_sig_iq
    if not _start_sig:
        _start_sig = samples_iq(_max_amp, 0) + samples_iq(-_max_amp, 0, 1) + samples_iq(_max_amp, 0, 2)
        I, Q = recover_iq(to_float_audio(_start_sig), freq, 0, iq_chunk_size = 1)
        _start_sig_iq = I + 1j * Q
        _start_sig_iq = np.flip(_start_sig_iq)
    return _start_sig
