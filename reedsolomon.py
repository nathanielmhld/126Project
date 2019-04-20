""" Reed-Solomon code """
import config
import unireedsolomon as rs
from huffman import _bitarr2bytes, _bytes2bitarr, _str2bitarr

class RSCode:
    """ Reed-Solomon encoder and decoder """
    def __init__(self, block_size = 0, block_content = 0):
        """ Create R.S. Code with given block size and given max message (content) length
            (block_content < block_size; corrects at most (block_content - block_size) / 2 errors) """
        self.block_size = block_size or config.RS_BLOCK_SIZE
        self.block_content = block_content or config.RS_BLOCK_CONTENT
        self.coder = rs.RSCoder(self.block_size, self.block_content)

    def encode(self, bitarr):
        output = []
        bitarr_bytes = _bitarr2bytes(bitarr)
        for i in range(0, len(bitarr_bytes), self.block_content):
            encoded = self.coder.encode(bitarr_bytes[i:i+self.block_content])
            output.extend(_str2bitarr(encoded))
        return output

    def decode(self, bitarr):
        output = []
        if len(bitarr) % (self.block_size * 8):
            bitarr = bitarr[:-(len(bitarr) % (self.block_size * 8))]
        bitarr_bytes = _bitarr2bytes(bitarr)
        for i in range(0, len(bitarr_bytes), self.block_size):
            try:
                decoded = self.coder.decode(bitarr_bytes[i:i+self.block_size])[0]
                output.extend(_str2bitarr(decoded))
            except:
                import sys
                print('Reed Solomon warning: partial decode failure', file=sys.stderr)
                output.extend(_bytes2bitarr(bitarr_bytes[i:i+self.block_content]))
        return output

    

