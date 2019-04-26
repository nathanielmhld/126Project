import pyaudio
import sys, os.path as path
import time, pickle
import numpy as np
from huffman_zlib import ZlibCoder
from huffman import HuffDict
from reedsolomon import RSCode
import config

# load huffDict
huffDict = HuffDict.from_save(config.HUFF_DICT_PATH)
zlibCoder = ZlibCoder()
rsCode = RSCode()

volume = 1.0     # range [0.0, 1.0]
fs = config.SAMPLING_RATE   # sampling rate, Hz, must be integer

p = pyaudio.PyAudio()
# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,#p.get_format_from_width(1),
                channels=1,
                rate=fs,
                output=True)

chunk_size = config.CHUNK_SIZE

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
#message = """The eras change, nations grow strong, or weaken, like Troy, magnificent in men and riches... And now displaying only ruins for wealth the old ancestral tombs. Sparta, Mycenae, Athens, and Thebes, all flourished once, and now what are they more than names? I hear that Rome Is rising..."""

#message = 'hello world!'

# from terminal argument
if len(sys.argv) > 1:
    message = sys.argv[1]
    # from file argument
    if len(sys.argv) > 2 and message == '-f':
        message = open(sys.argv[2], 'r').read()

#if config.DEBUG:
#    print('message is:')
#    print(message)

""" message should now be list of 0, 1 """
#message_binary = np.random.randint(2, size=(10000 * config.MESSAGE_BITS,)) # random message
if config.DEBUG:
    _ta1 = time.time()

zlib_message = zlibCoder[message]
huff_message = huffDict[message]
if config.DEBUG:
    _ta2 = time.time()
#print(config.RS_BLOCK_CONTENT)
print('input:', len(message) * 8, 'bits')
print('huff:', len(huff_message), 'bits')
print('zlib:', len(zlib_message), 'bits')
use_huff = len(huff_message) < len(zlib_message)

if use_huff:
    message_binary = rsCode.encode(huff_message)
else:
    message_binary = rsCode.encode(zlib_message)
if config.DEBUG:
    _ta3 = time.time()
print('rscode:', len(message_binary), 'bits')
#print(huff_message)
if config.DEBUG:
    # for debug, pass real message to receiver for computing BSC error rate
    np.save('_actual_message.npy', message_binary)
    test_rs_decode = rsCode.decode(message_binary)
    #print(len(test_rs_decode))
    #print(len(huff_message))
    if use_huff:
        assert test_rs_decode == huff_message
        test_huff_decode = huffDict[test_rs_decode]
        assert test_huff_decode.strip() == message.strip()
    else:
        assert test_rs_decode == zlib_message
        test_zlib_decode = zlibCoder[test_rs_decode]
        assert test_zlib_decode == message
    print('debug sanity checker: encoding working (able to decode locally)')
_last_played = [-1] * config.NUM_TRANSMITTERS

def samples(freqs, csize = effective_chunk_size, padl = standby_chunk_size, t = 0):
    """ generate samples, note conversion to float32 array """
    base_samps = np.arange(csize, dtype=np.float32) + t
    samps = np.zeros(base_samps.shape[0], dtype=np.float32)
    if freqs:
        total = 0.0
        for i, f in enumerate(freqs):
            samps_f = np.sin(2*np.pi*(base_samps)*f/ fs)
            samps += samps_f
            total += 1.
        samps = volume * samps / total
    samps_bytes = samps.tobytes()
    if padl:
        samps_bytes = b'\0\0\0\0' * padl + samps_bytes
    return samps_bytes

def encode_message(bits, i):
    """ encode the next len(CHANNEL_FREQS) bits into a message. i: current time step, for phase sync """
    all_freqs = []
    all_ids = []
    M = config.PACKET_BITS
    for t in range(config.NUM_TRANSMITTERS):
        ch_id = 0
        for j in range((i*2 + t)*M, min((i*2 + t + 1)*M, len(bits))):
            ch_id *= 2
            if bits[j] == ((j - (i*2 + t)*M) & 1):
                ch_id += 1
        chnl = config.TRANSMITTER_STARTS[t] + config.TRANSMITTER_INTERVALS[t] * ch_id
        if chnl == _last_played[t]:
            chnl = config.TRANSMITTER_CONTINUER[t]
        _last_played[t] = chnl
        all_freqs.append(chnl)
        all_ids.append(ch_id)
    if config.DEBUG:
        print(' '.join(map(str, all_freqs)), ':',
              ' '.join(map(str, all_ids)))
    return samples(all_freqs, t = i * config.MESSAGE_BITS)

samps_lst = []
for i, sfreqs in enumerate(config.START_SIGNAL):
    samps_lst.append(samples(sfreqs, chunk_size, 0) )

if config.DEBUG:
    _tb = time.time()
    print('init time:', _tb - _ta1)
    print('> compression:', _ta2 - _ta1)
    print('> EC code:', _ta3 - _ta2)
    print('> debug code:', _tb - _ta3)

# play. May repeat with different volume values (if done interactively)
for i in range(0, len(message_binary), config.MESSAGE_BITS):
    samps_lst.append(encode_message(message_binary, i // config.MESSAGE_BITS))

for i, sfreqs in enumerate(config.END_SIGNAL):
    samps_lst.append(samples(sfreqs, chunk_size, 0))
samps = b''.join(samps_lst)

print("\ntransmitting")
stream.write(samps)
print('finished transmitting, stopping')

stream.stop_stream()
stream.close()
p.terminate()
