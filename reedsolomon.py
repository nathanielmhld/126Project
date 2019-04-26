""" Reed-Solomon code """
import config
from reedsolo import RSCodec
from huffman import _bitarr2bytes, _bytes2bitarr, _str2bitarr, _unpad
import numpy as np

def _transpose(arr, blocksz = 0, nblocks = 0):
    tr = [0] * len(arr)
    if blocksz:
        nblocks = len(arr) // (blocksz * 8)
    elif nblocks:
        blocksz = len(arr) // (nblocks * 8)
    if blocksz * nblocks * 8 != len(arr):
        print('warning: transpose received invalid block size')
    for i in range(nblocks):
        for j in range(blocksz):
            for k in range(8):
                tr[j * nblocks * 8 + i * 8 + k] = arr[i * blocksz * 8 + j * 8 + k]
    return tr

class RSCode:
    """ Reed-Solomon encoder and decoder """
    def __init__(self, num_ec=None):
        """ Create R.S. Code with given number of EC symbols. corrects num_ec / 2 errors. """
        self.num_ec = num_ec or config.RS_NUM_EC
        self.coder = RSCodec(self.num_ec)

    def encode(self, bitarr):
        output = []
        bitarr_bytes = _bitarr2bytes(bitarr, 8, 8)
        encoded = self.coder.encode(bitarr_bytes)
        output = _bytes2bitarr(encoded)
        return output

    def decode(self, bitarr, starts_to_try=10):
        if not bitarr:
            print('warning: empty block received')
            return
        # sometimes we have extra bytes at the beginning, fix that by bruteforcing
        for offs in range(starts_to_try):
            try:
                if len(bitarr) % 8:
                    # cut off unaligned
                    bitarr_trim = bitarr[:-len(bitarr) % 8]
                else:
                    bitarr_trim = bitarr

                if config.DEBUG:
                    import os.path
                    decoded = bitarr_trim[:]
                    if os.path.exists('_actual_message.npy'):
                        if offs:
                            print('reed-solomon offset =', offs)
                        actual = np.load('_actual_message.npy')
                        while len(decoded) > len(actual):
                            decoded.pop()
                        while len(decoded) < len(actual):
                            decoded.append(0)
                        Y = np.array(decoded)
                        X = np.array(actual)
                        bitwise_errs = np.sum(np.abs(Y - X))
                        X = X.reshape(-1, 8)
                        Y = Y.reshape(-1, 8)
                        errs = np.sum(np.any(X != Y, axis=1))
                        #print(X)
                        #print(Y)
                        print('bit errors', bitwise_errs, '=', bitwise_errs / len(decoded))
                        print('byte errors', errs, '=', errs / (len(decoded) // 8))
                bitarr_bytes = _bitarr2bytes(bitarr_trim, None)
                decoded = self.coder.decode(bitarr_bytes)[0]
                output = _bytes2bitarr(decoded)
                return _unpad(output, 8)
            except:
                bitarr = bitarr[8:]
        raise Exception('FATAL: reed-solomon decoding failed')
