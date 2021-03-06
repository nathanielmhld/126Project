""" Reed-Solomon code """
import config
import unireedsolomon as rs
from huffman import _bitarr2bytes, _bytes2bitarr, _str2bitarr, _unpad

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
    def __init__(self, block_size = 0, block_content = 0, allow_partial_block = False):
        """ Create R.S. Code with given block size and given max message (content) length
            (block_content < block_size; corrects at most (block_content - block_size) / 2 errors)
            If allow_partial_block is set, then the last block may be made smaller to fit text size. """
        self.block_size = block_size or config.RS_BLOCK_SIZE
        self.block_content = block_content or config.RS_BLOCK_CONTENT
        self.coder = rs.RSCoder(self.block_size, self.block_content)
        self.allow_partial_block = allow_partial_block

    def encode(self, bitarr):
        output = []
        if self.allow_partial_block:
            bitarr_bytes = _bitarr2bytes(bitarr, 8)
        else:
            bitarr_bytes = _bitarr2bytes(bitarr, self.block_content * 8)
        #print('e', len(bitarr_bytes))
        for i in range(0, len(bitarr_bytes), self.block_content):
            print(i)
            input_bytes = bitarr_bytes[i:i+self.block_content]
            if self.allow_partial_block and len(input_bytes) * 2 < self.block_content: 
                partial_block_size = len(input_bytes) * 2
                partial_block_coder = rs.RSCoder(partial_block_size, len(input_bytes))
                encoded = partial_block_coder.encode(input_bytes)
            else:
                encoded = self.coder.encode(input_bytes)
            output.extend(_str2bitarr(encoded))
        if not self.allow_partial_block:
            output_tr = _transpose(output, blocksz = self.block_size)
        else:
            output_tr = output
        return output_tr

    def decode(self, bitarr):
        if not bitarr:
            print('warning: empty block received')
            return
        output = []
        if not self.allow_partial_block and len(bitarr) % (self.block_size * 8):
            # cut off unaligned
            bitarr = bitarr[:-(len(bitarr) % (self.block_size * 8))]
        if self.allow_partial_block:
            bitarr_tr = bitarr
        else:
            bitarr_tr = _transpose(bitarr, nblocks = self.block_size)
        bitarr_bytes = _bitarr2bytes(bitarr_tr, False)
        fail = False

        for op in range(3):
            fail = False
            for i in range(0, len(bitarr_bytes), self.block_size):
                try:
                    enc_bytes = bitarr_bytes[i:i+self.block_size]
                    if self.allow_partial_block and len(enc_bytes) < self.block_size:
                        partial_block_coder = rs.RSCoder(len(enc_bytes), len(enc_bytes)//2)
                        decoded = partial_block_coder.decode(enc_bytes)[0]
                    else:
                        decoded = self.coder.decode(enc_bytes)[0]
                    if len(decoded) < self.block_content:
                        diff = self.block_content - len(decoded)
                        decoded = '\0' * diff + decoded
                    #print('d', _str2bitarr(decoded))
                    output.extend(_str2bitarr(decoded))
                except:
                    fail = True
                    output.extend(_bytes2bitarr(bitarr_bytes[i:i+self.block_content]))
            if not fail:
                break
            # hardcoded off-by-one fixes
            if op == 0:
                # try adding a byte at beginning
                bitarr_bytes = b'\0' + bitarr_bytes 
            elif op == 1:
                # try deleting a byte at end
                bitarr_bytes = bitarr_bytes[1:-1]

        return _unpad(output)

    

