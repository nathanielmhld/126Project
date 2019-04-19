SAMPLING_RATE = 44100 # sampling rate, Hz, must be integer

# bit rate is 1/MESSAGE_DURATION * FREQ_CHANNELS
MESSAGE_DURATION = 0.2 # time per message
FREQ_CHANNELS = 20 # number of distinct frequency channels
LOW_FREQ = 300.0 # minimum frequency
FREQ_INTERVAL = 150.0 # interval between channels
FREQ_THRESH = 10.0
AMP_THRESH = 2.0
RS_BLOCK_SIZE = 16
RS_BLOCK_CONTENT = 8
