DEBUG = False # if false, turns off useless prints

SAMPLING_RATE = 44100 # sampling rate, Hz, must be integer

# bit rate is 1/MESSAGE_DURATION * FREQ_CHANNELS

# transmitter config
PACKET_TIME = 0.047 # time per packet
PACKET_WAIT_TIME = 0.017 # time spent on 'standby' per packet
PACKET_BITS = 4 # number of bits per packet per transmitter

TRANSMITTER_STARTS = [1050, 2380] # start freqs for each transmitter
TRANSMITTER_INTERVALS = [60, 90] # freq interval for each transmitter
TRANSMITTER_CONTINUER = [2070, 3900] # the special 'continue last symbol' frequencies

# stft steps per chunk
STFT_STEP = 4

NUM_TRANSMITTERS = len(TRANSMITTER_STARTS) # number of transmitters (autofill)
MESSAGE_BITS = PACKET_BITS * NUM_TRANSMITTERS # number of bits in total per message (autofill)
CHUNK_SIZE = int(PACKET_TIME * SAMPLING_RATE)

# ensure divisible
if CHUNK_SIZE % STFT_STEP:
    CHUNK_SIZE += STFT_STEP - CHUNK_SIZE % STFT_STEP

# frequency bin for start/end signals - freqs in +- this value are binned together
FREQ_THRESH = 20.0

# cost to deviate from default decoder alignment by one bit, used in decoder alignment stage
DECODER_ALIGNMENT_FINE_TUNE_COST = 0.005

# reed-solomon: corrects (block_size - block_content) / 2
RS_NUM_EC = 76

# deprecated options
# RS_BLOCK_SIZE = 32
# RS_BLOCK_CONTENT = 24
# RS_ALLOW_PARTIAL_BLOCK = False # if yes, saves space by using variable-size RS blocks; ignores RS_BLOCK_SIZE!

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
               [4200],
               [4200],
               [4200],
               [4200],
               [4200],
               [4200],
               [4200],
             ]

# noise reference
REF_FREQS = [50, 110, 220, 330, 440, 660, 770, 880, 990, 1000, 1500]

import os.path
# project root dir
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# huff dict path
HUFF_DICT_PATH = os.path.join(ROOT_DIR, "huffman_model.pkl")
