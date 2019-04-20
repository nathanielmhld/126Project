SAMPLING_RATE = 44100 # sampling rate, Hz, must be integer

# bit rate is 1/MESSAGE_DURATION * FREQ_CHANNELS
MESSAGE_DURATION = 0.1 # time per message
MESSAGE_BITS = 8 # number of bits per message
LOW_FREQ = 320.0 # minimum frequency
FREQ_INTERVAL = 12.0 # interval between channels

CHANNEL_FREQS = [400, 700, 1000,  1300, 1600, 1900, 2200, 2600]#, 3100, 3600]
CHANNEL_POW =   [4.,  1.5, 1.0,  1.0,  1.0,  1.0,  1.0,  1.0]#,  1.5,  2.0]

FREQ_THRESH = 10.0
AMP_THRESH = 0.5

RS_BLOCK_SIZE = 128
RS_BLOCK_CONTENT = 80
START_SIGNAL = [
               # [650, 975, 1950],
                 [3392, 3600],
               ]

END_SIGNAL = [
               [3450, 3550],
             ]
