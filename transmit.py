import pyaudio
import numpy as np
from huffman import HuffDict
from reedsolomon import RSCode
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode(allow_partial_block = config.RS_ALLOW_PARTIAL_BLOCK)

p = pyaudio.PyAudio()

volume = 1.0     # range [0.0, 1.0]
fs = config.SAMPLING_RATE   # sampling rate, Hz, must be integer

chunk_size = int(config.MESSAGE_DURATION * config.SAMPLING_RATE)

# Liber Primus by cicada
message = """Welcome, pilgrim, to the great journey toward the end of all things. It is not an easy trip, but for those who find their way here it is a necessary one. Along the way you will find an end to all struggle and suffering, your innocence, your illusions, your certainty, and your reality. Ultimately, you will discover an end to self.
"""
"""
It is through this pilgrimage that we shape ourselves and our realities. Journey deep within and you will arrive outside. Like the instar, it is only through going within that we may emerge.
Wisdom: you are a being unto yourself. You are a law unto yourself. Each intelligence is holy. For all that lives is holy. An instruction: command your own self.
Some wisdom: the primes are sacred. The totient function is sacred. All things should be encrypted, know this."""

# Pythagora's speech, from Virgil
message = """The eras change, nations grow strong, or weaken, like Troy, magnificent in men and riches... And now displaying only ruins for wealth the old ancestral tombs. Sparta, Mycenae, Athens, and Thebes, all flourished once, and now what are they more than names? I hear that Rome Is rising..."""

#message = 'hello world!'

""" message should now be list of 0, 1 """
#message_binary = np.random.randint(2, size=(10000 * config.MESSAGE_BITS,)) # random message
message_binary = huffDict[message]
#print(len(message_binary))
#print(config.RS_BLOCK_CONTENT)
print('input:', len(message) * 8, 'bits')
print('huff:', len(message_binary), 'bits')
message_binary = rsCode.encode(message_binary)
print('rscode:', len(message_binary), 'bits')

# for fake decoder
decoded = []

def samples(freqs, t=0):
    """ generate samples, note conversion to float32 array """
    base_samps = np.arange(chunk_size, dtype=np.float32)
    samps = np.zeros(base_samps.shape[0], dtype=np.float32)
    #freqs=[200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    if freqs:
        total = 0.0
        for i, f in enumerate(freqs):
            samps_f = np.sin(2*np.pi*(base_samps + t * chunk_size)*f/ fs)
            weight = 1.
            samps += samps_f * weight
            total += weight
        samps = volume * samps / total

        # TODO: write to file
    # fake decoder (does fft on samples) for debug
    if t >= 0:
        #freq_channels = config.CHANNEL_FREQS
        fake_input = samps.copy()
        #if np.random.random() < 0.12:
            # simulate burst: completely destroy
            #fake_input = np.zeros(*fake_input.shape)
            #print(' (zeroed) ', end='')
        # simulate random distortion
        #fake_input += np.random.randn(*fake_input.shape) * 0.1
        data_fft = np.fft.rfft(fake_input, norm="ortho")
        mags = np.abs(data_fft)
        freqs = fs * np.fft.rfftfreq(len(samps))
        #ints = [0.] * config.MESSAGE_BITS

        high_freq = freqs[np.argmax(mags)] 
        recon = int(np.round((high_freq - config.LOW_FREQ) / config.FREQ_INTERVAL))
        print("The highest frequency is {} Hz, {}".format(high_freq, recon))

        digit = (1 << (config.MESSAGE_BITS-1))
        i = config.MESSAGE_BITS-1
        while digit:
            decoded.append(((recon & digit) >> i) ^ ((config.MESSAGE_BITS-i) & 1))
            digit >>= 1
            i -= 1
        print(' fake decoder:', decoded[-config.MESSAGE_BITS:], end=' ')
        print()

    #print(samps)

    #samps = (samps * 127 + 128).astype(np.int)
    #samps = ''.join([chr(x) for x in samps])
    # i = len(samps)
    # while abs(samps[i-1]) > 1e-12 or samps[i-2] > 0:
        # i -= 1
    return samps.tobytes()#[:i]

# [650, 975, 1950],
def encode_to_frequency(bits, i):
    """ encode the next len(CHANNEL_FREQS) bits into frequencies """
    all_freqs = []
    M = config.MESSAGE_BITS
    ch_id = 0
    for j in range(i*M, min((i+1)*M, len(bits))):
        #f1 = config.CHANNEL_FREQS[j-i*M]
        ch_id *= 2
        if bits[j] == ((j - i*M) & 1):
            ch_id += 1
            #all_freqs.append(f1)
    print(ch_id)
    all_freqs = [config.LOW_FREQ + config.FREQ_INTERVAL * ch_id]
    print('playing: ', '\t'.join(map(str, all_freqs)))

    return samples(all_freqs, i)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,#p.get_format_from_width(1),
                channels=1,
                rate=fs,
                output=True)

print("syncronizing")
for i, sfreqs in enumerate(config.START_SIGNAL):
    stream.write(samples(sfreqs, -len(config.START_SIGNAL)+i))
print("\ntransmitting")

# play. May repeat with different volume values (if done interactively)
for i in range(0, len(message_binary), config.MESSAGE_BITS):
    stream.write(encode_to_frequency(message_binary, i // config.MESSAGE_BITS))

test_rs_decode = rsCode.decode(decoded)
test_decode = huffDict[test_rs_decode]
print('bits received:', len(decoded))
print('fake decoder got:', test_decode)

print("\ntransmitting end signal")
for i, sfreqs in enumerate(config.END_SIGNAL):
    stream.write(samples(sfreqs, -1))
print('finished transmitting, stopping')

stream.stop_stream()
stream.close()

p.terminate()
