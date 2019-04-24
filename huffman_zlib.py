"""
zlib interface, as replacement for our own Huffman code
"""

import zlib
from huffman import _bytes2bitarr, _bitarr2bytes

class HuffDict:
    @staticmethod
    def from_save(path):
        """ does nothing """
        return HuffDict()
    def __getitem__(self, key):
        """ encode string => code, OR decode code (list/tuple/int) => string """
        if type(key) is str:
            return self.encode(key)
        elif type(key) is list or type(key) is tuple:
            return self.decode(key)
    def encode(self, input):
        return _bytes2bitarr(zlib.compress(input.encode('ascii'), level=9))
    def decode(self, input):
        return zlib.decompress(_bitarr2bytes(input)).decode('ascii')


