import pyaudio
import numpy as np
from huffman import HuffDict
from reedsolomon import RSCode
import math, time
import bpsk
import config

# load huffDict
huffDict = HuffDict.from_save('huffman_model.pkl')
rsCode = RSCode(allow_partial_block = config.RS_ALLOW_PARTIAL_BLOCK)

p = pyaudio.PyAudio()

from bpsk import fs, chunk_size
print('chunk size:', chunk_size)

# Liber Primus by cicada
message = """Welcome, pilgrim, to the great journey toward the end of all things. It is not an easy trip, but for those who find their way here it is a necessary one. Along the way you will find an end to all struggle and suffering, your innocence, your illusions, your certainty, and your reality. Ultimately, you will discover an end to self.
"""
"""
It is through this pilgrimage that we shape ourselves and our realities. Journey deep within and you will arrive outside. Like the instar, it is only through going within that we may emerge.
Wisdom: you are a being unto yourself. You are a law unto yourself. Each intelligence is holy. For all that lives is holy. An instruction: command your own self.
Some wisdom: the primes are sacred. The totient function is sacred. All things should be encrypted, know this."""

# Pythagora's speech, from Virgil
#message = """The eras change, nations grow strong, or weaken, like Troy, magnificent in men and riches... And now displaying only ruins for wealth the old ancestral tombs. Sparta, Mycenae, Athens, and Thebes, all flourished once, and now what are they more than names? I hear that Rome Is rising..."""

message = 'hello world!'

def samples_multifreq(freqs, t=0):
    """ generate samples with given frequencies, note conversion to float32 array """
    base_samps = np.arange(chunk_size, dtype=np.float32)
    samps = np.zeros(base_samps.shape[0], dtype=np.float32)
    if freqs:
        total = 0.0
        for i, f in enumerate(freqs):
            samps_f = np.cos(2*np.pi*(base_samps + t * chunk_size)*f/ fs)
            weight = 1.
            samps += samps_f * weight
            total += weight
        samps = config.VOLUME * samps / total
    return samps.tobytes()


# [650, 975, 1950],
if __name__ == '__main__':
    #message_binary = np.random.randint(2, size=(10000 * config.MESSAGE_BITS,)) # random message
    print("\nencoding message")
    message_binary = huffDict[message]
    #print(len(message_binary))
    #print(config.RS_BLOCK_CONTENT)
    print('input:', len(message) * 8, 'bits')
    print('huff:', len(message_binary), 'bits')
    message_binary = rsCode.encode(message_binary)
    print('rscode:', len(message_binary), 'bits')

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,#p.get_format_from_width(1),
                    channels=1,
                    rate=fs,
                    output=True)

    t1 = time.time()
    samps = b''
    # start signal for receiver to get ready
    for i, sfreqs in enumerate(config.START_SIGNAL):
        for _ in range(max(int(0.4 / config.MESSAGE_DURATION), 1)):
            samps += samples_multifreq(sfreqs, -1)

    print('start signal len:', len(samps)//4)
    # bpsk start/alignment signal
    samps += bpsk.start_sig()

    # content
    message_binary = np.random.randint(2, size=(len(message_binary),)) # random message
    print(message_binary)
    for i in range(0, len(message_binary), config.MESSAGE_BITS):
        samps += bpsk.encode(message_binary, i // config.MESSAGE_BITS)
    print('audio:', len(samps)//4, 'samples')

    # end signal for receiver to stop automatically
    for i, sfreqs in enumerate(config.END_SIGNAL):
        for _ in range(max(int(1.8 / config.MESSAGE_DURATION), 1)):
            samps += samples_multifreq(sfreqs, -1)
    t2 = time.time()

    print("\ntransmitting")
    stream.write(samps)
    t3 = time.time()

    print('all done. stats:' )
    print('total time:',t3 - t1, 'encoding:', t2 - t1, 'transmitting:', t3 - t2)
    print('channel bitrate:', len(message_binary) / (t3 - t1))
    print('real bitrate (incl. compression, ecc):', len(message) * 8 / (t3 - t1))
    print()
    
    print('debug: fake decoder working...' )
    audio = bpsk.to_float_audio(samps)

    import matplotlib.pyplot as plt

    freq = config.CARRIER_FREQ
    vol = np.absolute(bpsk.fft_freq(audio, freq))
    sqr = np.square(audio/vol)

    pstart, pend = 17200, 17400

    binsz = chunk_size // 6
    #mag, ang = bpsk.stft_freq(audio, freq, binsz, 1, pstart, pend+binsz-1)
    mag_sqr, ang_sqr = bpsk.stft_freq(sqr, freq*2, binsz, 1, pstart, pend+binsz-1)

    bt, breaks = [], []
    for i, ang in enumerate(ang_sqr):
        t = (i + pstart) / fs
        rang = -ang
        if ang > 0:
            rang = np.pi * 2 - ang
        breaks.append(t + rang / (2 * np.pi) / (freq*2))
        #breaks.append(t + rang / (2 * np.pi) / freq)
        bt.append(2)

    x_axis = np.linspace(pstart / fs, pend / fs, pend - pstart)
    plt.figure()
    plt.plot(x_axis, audio[pstart:pend], label='sig')
    plt.plot(x_axis, sqr[pstart:pend], label='sqr')
    plt.plot(x_axis, ang_sqr, label='sqr_conv_ang')
    plt.scatter(breaks, bt, c='b', label='sig_start')
    plt.legend()
    plt.show()
    import sys
    sys.exit(0)

    decoded = bpsk.decode(audio, config.CARRIER_FREQ);
    #print(decoded)
    #print(message_binary)
    #print(decoded)
    
    #r = stft(audio)

    test_rs_decode = rsCode.decode(decoded)
    test_decode = huffDict[test_rs_decode]
    print('bits received:', len(decoded))
    print('fake decoder got:', test_decode)

    stream.stop_stream()
    stream.close()

p.terminate()
