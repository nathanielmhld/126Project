import pyaudio
import numpy as np
from huffman import HuffDict
from reedsolomon import RSCode
import config

# load huffDictj
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode()

p = pyaudio.PyAudio()

volume = 1.0     # range [0.0, 1.0]
fs = config.SAMPLING_RATE   # sampling rate, Hz, must be integer

chunk_size = int(config.MESSAGE_DURATION * config.SAMPLING_RATE)

message = """Welcome, pilgrim, to the great journey toward the end of all things. It is not an easy trip, but for those who find their way here it is a necessary one. Along the way you will find an end to all struggle and suffering, your innocence, your illusions, your certainty, and your reality ultimately, you will discover an end to self."""

#message = "hello world m"

""" message should now be list of 0, 1 """
#message_binary = np.random.randint(2, size=(10000 * config.FREQ_CHANNELS,)) # random message
message_binary = huffDict[message]
print(message_binary)
message_binary = rsCode.encode(message_binary)

# for fake decoder
decoded = []

def samples(freqs, t=0):
    """ generate samples, note conversion to float32 array """
    base_samps = np.arange(chunk_size, dtype=np.float32)*1.5 
    samps = np.zeros(base_samps.shape[0], dtype=np.float32)
    #freqs=[200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    if freqs:
        total = 0.0
        for f in freqs:
            samps_f = np.sin(2*np.pi*base_samps*f/ fs)
            weight = 1. #/ f**1.4
            samps += samps_f * weight
            total += weight
        samps = volume * samps / total

        # TODO: write to file
    # fake decoder (does fft on samples) for debug

    freq_channels = np.arange(config.LOW_FREQ, config.LOW_FREQ + config.FREQ_INTERVAL * config.FREQ_CHANNELS, config.FREQ_INTERVAL)
    data_fft = np.fft.rfft(samps, norm="ortho")
    mags = np.abs(data_fft)
    freqs = fs * 2./3 * np.fft.rfftfreq(len(samps))
    #plt.plot(freqs)
    ints = [0.] * config.FREQ_CHANNELS
    #print(freqs)
    j = 0
    for i, f in enumerate(freqs):
        if abs(f - freq_channels[j]) < config.FREQ_THRESH:
            ints[j] += mags[i] / config.FREQ_THRESH / len(freqs) * fs
        elif f > freq_channels[j] and j < config.FREQ_CHANNELS-1:
            j += 1
    for i in range(config.FREQ_CHANNELS):
        if ints[i] > config.AMP_THRESH:
            decoded.append(1)
        else:
            decoded.append(0)

    print(' fake decoder:', decoded[-config.FREQ_CHANNELS:], end=' ')
    print()

    #print(samps)

    samps = (samps * 127 + 128).astype(np.int)
    samps = ''.join([chr(x) for x in samps])
    # i = len(samps)
    # while abs(samps[i-1]) > 1e-12 or samps[i-2] > 0:
        # i -= 1
    return samps#[:i]

def encode_to_frequency(bits, i):
    """ encode the next FREQ_CHANNELS bits into frequencies """
    f1 = config.LOW_FREQ
    all_freqs = []
    for j in range(i*config.FREQ_CHANNELS, min((i+1)*config.FREQ_CHANNELS, len(bits))):
        if bits[j]:
            all_freqs.append(f1)
        f1 += config.FREQ_INTERVAL
    print('playing: ', '\t'.join(map(str, all_freqs)))
    return samples(all_freqs, i)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=p.get_format_from_width(1),
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
for i in range(0, len(message_binary), config.FREQ_CHANNELS):
    stream.write(encode_to_frequency(message_binary, i // config.FREQ_CHANNELS))

#decoded = decoded[:len(message_binary)]
test_decode = huffDict[rsCode.decode(decoded)]
print('fake decoded got: ', test_decode)
print('finished, stopping')

stream.stop_stream()
stream.close()

p.terminate()
