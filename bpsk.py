'''
BPSK: Binary Phase Shift Keying tools
to be used instead of QAM for better noise tolerance.
'''
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

def stft_freq(audio, freq, binsz, interval, tstart=0, tend=-1):
    ''' single frequency short-time fourier transform '''
    if tend == -1:
        tend = audio.size
    lrange = range(tstart, tend-binsz+1, interval)
    ret = np.zeros(len(lrange), dtype=np.complex)
    for t, i in enumerate(lrange):
        ret[t] = fft_freq(audio, freq, i, i + binsz)# np.sum(conv[i:i+binsz])
    return np.absolute(ret), np.angle(ret)

_window = np.hstack([np.arange(0, chunk_size // 2, dtype=np.float32), \
                       np.arange(chunk_size - chunk_size // 2, 0, -1, dtype=np.float32)])
_window /= np.max(_window)
_window = np.sqrt(_window)

def samples_phase(bit, freq = config.CARRIER_FREQ):
    """ generate phase-encoded samples and carrier freq, note conversion to float32 array """
    #inphase = 0.25
    #quad = 0.25
    base_samps = np.arange(chunk_size, dtype=np.float32)
    base_samps = np.fmod(base_samps, fs / freq)
    samps = np.cos(2*np.pi*base_samps* freq / fs + np.pi * (1 - bit))
    samps *= config.VOLUME
    #samps *= _window

    return samps.tobytes()

def encode(bits, i):
    """ encode the next bit """
    #print('playing: ', inphase, quad, i, bits[i*config.MESSAGE_BITS:(i+1)*config.MESSAGE_BITS])
    return samples_phase(bits[i * config.MESSAGE_BITS])

def decode(audio, freq, begin = -1):
    ''' recover binary signal from audio, use begin = -1 to auto find start signal '''
    if begin < 0:
        begin = find_start(audio, freq)

    vol = np.absolute(fft_freq(audio, freq))
    sqr = np.square(audio / vol)

    binsz = 1000
    for i in range(0, sqr.size-binsz+1, 10):
        conv = fft_freq(audio, freq, i, i + binsz)
        conv_sqr = fft_freq(sqr, freq*2, i, i + binsz)
        print(i, np.angle(conv), np.absolute(conv), np.angle(conv_sqr), np.absolute(conv_sqr), np.angle(conv) - np.angle(conv_sqr))
    """
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
    """
    return []

def find_start(audio, freq):
    return 0;

_start_sig = None
def start_sig():
    global _start_sig
    if not _start_sig:
        _start_sig = samples_phase(0) + samples_phase(1) + samples_phase(0)
    return _start_sig
