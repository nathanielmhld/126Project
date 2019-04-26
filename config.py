DEBUG = False # if false, turns off useless prints

SAMPLING_RATE = 44100 # sampling rate, Hz, must be integer

# bit rate is 1/MESSAGE_DURATION * FREQ_CHANNELS

# transmitter config
PACKET_TIME = 0.06 # time per packet
PACKET_WAIT_TIME = 0.017 # time spent on 'standby' per packet
PACKET_BITS = 4 # number of bits per packet per transmitter

TRANSMITTER_STARTS = [1050, 2300] # start freqs for each transmitter
TRANSMITTER_INTERVALS = [50, 90] # freq interval for each transmitter
TRANSMITTER_CONTINUER = [1900, 3830] # the special 'continue last symbol' frequencies

# stft steps per chunk
STFT_STEP = 8

NUM_TRANSMITTERS = len(TRANSMITTER_STARTS) # number of transmitters (autofill)
MESSAGE_BITS = PACKET_BITS * NUM_TRANSMITTERS # number of bits in total per message (autofill)
CHUNK_SIZE = int(PACKET_TIME * SAMPLING_RATE)

# ensure divisible
if CHUNK_SIZE % STFT_STEP:
    CHUNK_SIZE += STFT_STEP - CHUNK_SIZE % STFT_STEP

# deprecated
LOW_FREQ = 320.0 # minimum frequency
FREQ_INTERVAL = 12.0 # interval between channels

FREQ_THRESH = 10.0
AMP_THRESH = 0.5

# reed-solomon: corrects (block_size - block_content) / 2
RS_BLOCK_SIZE = 64
RS_BLOCK_CONTENT = 50
RS_ALLOW_PARTIAL_BLOCK = False # if yes, saves space by using variable-size RS blocks, very bittle!

# start signal design
START_SIGNAL = [
                 [4300],
                 [4300],
                 [4300],
                 [4300],
                 [4300],
                 [4300],
               ]

# end signal design
END_SIGNAL = [
               [4200],
               [4200],
               [4200],
               [4200],
               [4200],
               [4200],
               [4200],
             ]
