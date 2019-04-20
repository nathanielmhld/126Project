""" Reed-Solomon code """
import config
import unireedsolomon as rs
from huffman import _bitarr2bytes, _bytes2bitarr, _str2bitarr, _unpad

class RSCode:
    """ Reed-Solomon encoder and decoder """
    def __init__(self, block_size = 0, block_content = 0, allow_partial_block = True):
        """ Create R.S. Code with given block size and given max message (content) length
            (block_content < block_size; corrects at most (block_content - block_size) / 2 errors)
            If allow_partial_block is set, then the last block may be made smaller to fit text size. """
        self.block_size = block_size or config.RS_BLOCK_SIZE
        self.block_content = block_content or config.RS_BLOCK_CONTENT
        self.coder = rs.RSCoder(self.block_size, self.block_content)
        self.allow_partial_block = allow_partial_block

    def encode(self, bitarr):
        output = []
        bitarr_bytes = _bitarr2bytes(bitarr)
        for i in range(0, len(bitarr_bytes), self.block_content):
            input_bytes = bitarr_bytes[i:i+self.block_content]
            if self.allow_partial_block and len(input_bytes) * 2 < self.block_content: 
                partial_block_size = len(input_bytes) * 2
                partial_block_coder = rs.RSCoder(partial_block_size, len(input_bytes))
                encoded = partial_block_coder.encode(input_bytes)
            else:
                encoded = self.coder.encode(input_bytes)
            output.extend(_str2bitarr(encoded))
        return output

    def decode(self, bitarr):
        output = []
        if not self.allow_partial_block and len(bitarr) % (self.block_size * 8):
            # cut off unaligned
            bitarr = bitarr[:-(len(bitarr) % (self.block_size * 8))]
        bitarr_bytes = _bitarr2bytes(bitarr, False)
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

    

