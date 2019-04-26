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
FREQ_THRESH = 10.0

# reed-solomon: corrects (block_size - block_content) / 2
RS_NUM_EC = 66

# deprecated options
RS_BLOCK_SIZE = 32
RS_BLOCK_CONTENT = 24
RS_ALLOW_PARTIAL_BLOCK = False # if yes, saves space by using variable-size RS blocks; ignores RS_BLOCK_SIZE!

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
               [4500],
               [4500],
               [4500],
               [4500],
               [4500],
               [4500],
               [4500],
             ]

# huff dict path
HUFF_DICT_PATH = "huffman_model.pkl"
