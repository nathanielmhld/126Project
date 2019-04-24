SAMPLING_RATE = 44100 # sampling rate, Hz, must be integer

# transmitter options
MESSAGE_DURATION = 0.06 # time per message
#MESSAGE_BITS = 4 # number of bits per message, we are using 2**x-QAM. must be multiple of 2
MESSAGE_BITS = 1 # number of bits per message, must be 1 for BPSK right now.
CARRIER_FREQ = 1800 # carrier frequency
REF_FREQS = [880, 1760] # reference frequencies for testing background noise

#LOW_FREQ = 320.0 # minimum frequency
#FREQ_INTERVAL = 12.0 # interval between channels

#CHANNEL_FREQS = [400, 700, 1000,  1300, 1600, 1900, 2200, 2600]#, 3100, 3600]
#CHANNEL_POW =   [4.,  1.5, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]#,  1.5,  2.0]

VOLUME = 0.5     # range [0.0, 1.0]

FREQ_THRESH = 5.0
AMP_THRESH = 0.5


RS_BLOCK_SIZE = 64
RS_BLOCK_CONTENT = 40
RS_ALLOW_PARTIAL_BLOCK = False
START_SIGNAL = [
                 [3395, 3605],
               ]

END_SIGNAL = [
               [4005, 4405],
             ]
