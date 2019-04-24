import pyaudio
import sys
import numpy as np
from huffman_zlib import HuffDict
from reedsolomon import RSCode
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode(allow_partial_block = config.RS_ALLOW_PARTIAL_BLOCK)

p = pyaudio.PyAudio()

volume = 1.0     # range [0.0, 1.0]
fs = config.SAMPLING_RATE   # sampling rate, Hz, must be integer

chunk_size = int(config.PACKET_TIME * config.SAMPLING_RATE)
standby_chunk_size = int(config.PACKET_WAIT_TIME * config.SAMPLING_RATE)
effective_chunk_size = chunk_size - standby_chunk_size

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

# from terminal argument
if len(sys.argv) > 1:
    message = sys.argv[1]
    # from file argument
    if len(sys.argv) > 2 and message == '-f':
        message = open(sys.argv[2], 'r').read()

""" message should now be list of 0, 1 """
#message_binary = np.random.randint(2, size=(10000 * config.MESSAGE_BITS,)) # random message
message_binary = huffDict[message]
#print(config.RS_BLOCK_CONTENT)
print('input:', len(message) * 8, 'bits')
print('huff:', len(message_binary), 'bits')
message_binary = rsCode.encode(message_binary)
print('rscode:', len(message_binary), 'bits')

_last_played = [-1] * config.NUM_TRANSMITTERS
def samples(freqs, t=0, has_standby = True):
    """ generate samples, note conversion to float32 array """
    csize = effective_chunk_size if has_standby else chunk_size
    base_samps = np.arange(csize, dtype=np.float32) + t * csize
    samps = np.zeros(base_samps.shape[0], dtype=np.float32)
    #freqs=[200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
    if freqs:
        total = 0.0
        for i, f in enumerate(freqs):
            samps_f = np.sin(2*np.pi*(base_samps)*f/ fs)
            weight = 1.
            samps += samps_f * weight
            total += weight
        samps = volume * samps / total
    if has_standby:
        return b'\0\0\0\0' * standby_chunk_size + samps.tobytes()
    else:
        return samps.tobytes()

def encode_message(bits, i):
    """ encode the next len(CHANNEL_FREQS) bits into a message. i: current time step, for phase sync """
    all_freqs, all_ids = [], []
    M = config.PACKET_BITS
    for t in range(config.NUM_TRANSMITTERS):
        ch_id = 0
        for j in range((i*2 + t)*M, min((i*2 + t + 1)*M, len(bits))):
            #f1 = config.CHANNEL_FREQS[j-i*M]
            ch_id *= 2
            if bits[j] == ((j - (i*2 + t)*M) & 1):
                ch_id += 1
                #all_freqs.append(f1)
        chnl = config.TRANSMITTER_STARTS[t] + config.TRANSMITTER_INTERVALS[t] * ch_id
        if chnl == _last_played[t]:
            chnl = config.TRANSMITTER_CONTINUER[t]
        _last_played[t] = chnl
        all_freqs.append(chnl)
        all_ids.append(ch_id)
    if config.DEBUG:
        print(' '.join(map(str, all_freqs)), ':',
              ' '.join(map(str, all_ids)))
    return samples(all_freqs, i)

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,#p.get_format_from_width(1),
                channels=1,
                rate=fs,
                output=True)

samps = b''
for i, sfreqs in enumerate(config.START_SIGNAL):
    samps += samples(sfreqs, -len(config.START_SIGNAL)+i, has_standby=False) 
# play. May repeat with different volume values (if done interactively)
for i in range(0, len(message_binary), config.MESSAGE_BITS):
    samps += encode_message(message_binary, i // config.MESSAGE_BITS)

for i, sfreqs in enumerate(config.END_SIGNAL):
    samps += samples(sfreqs, -1, has_standby=False)

print("\ntransmitting")
stream.write(samps)
print('finished transmitting, stopping')

stream.stop_stream()
stream.close()
p.terminate()
