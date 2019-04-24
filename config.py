DEBUG = False # if false, turns off useless prints

SAMPLING_RATE = 44100 # sampling rate, Hz, must be integer

# bit rate is 1/MESSAGE_DURATION * FREQ_CHANNELS

# transmitter config
PACKET_TIME = 0.07 # time per packet
PACKET_WAIT_TIME = 0.02 # time spent on 'standby' per packet
PACKET_BITS = 4 # number of bits per packet per transmitter

TRANSMITTER_STARTS = [1050, 2300] # start freqs for each transmitter
TRANSMITTER_INTERVALS = [50, 70] # freq interval for each transmitter
TRANSMITTER_CONTINUER = [1900, 3490] # the special 'continue last symbol' frequencies

NUM_TRANSMITTERS = len(TRANSMITTER_STARTS) # number of transmitters (autofill)
MESSAGE_BITS = PACKET_BITS * NUM_TRANSMITTERS # number of bits in total per message (autofill)

# deprecated
LOW_FREQ = 320.0 # minimum frequency
FREQ_INTERVAL = 12.0 # interval between channels

FREQ_THRESH = 5.0
AMP_THRESH = 0.5

# reed-solomon: corrects (block_size - block_content) / 2
RS_BLOCK_SIZE = 32
RS_BLOCK_CONTENT = 22
RS_ALLOW_PARTIAL_BLOCK = False # if yes, saves space by using variable-size RS blocks, very bittle!

# start signal design
START_SIGNAL = [
                 [4600],
                 [4600],
                 [4600],
                 [4600],
                 [4600],
                 [4600],
               ]

# end signal design
END_SIGNAL = [
               [4000],
               [4000],
               [4000],
               [4000],
               [4000],
               [4000],
               [4000],
             ]
