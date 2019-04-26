"""
methods for Huffman coding
To build HuffDict: x = HuffDict('sample_english.txt', 2); x.save('out_file.pkl')
To load saved HuffDict: x = HuffDict.from_save('out_file.pkl')
To encode: x[string] OR x.encode(string)
To decode: x[encoded_sequence] OR x.decode(encoded_sequence)
WARNING: may pad spaces at the end if length is not a multiple of n (which is 2 in our current model)
"""
import config
import numpy as np

def _s2i_128(input, i=0, n=-1):
    """ helper for converting string to integer using ascii starting at position i, using n chars per entity """
    if n == -1:
        n = len(input)
    output = 0
    for j in range(i, min(i+n, len(input))):    
        output <<= 7
        char_id = ord(input[j])
        if char_id < 0 or char_id >= 128:
            char_id = ord(' ')
        output += char_id
    return output

def _i2s_128(input, i=0, n=-1):
    """ helper for converting int to string (inverse of _s2i) """
    output = ''
    while input:
        output += chr(input & 0x7f)
        input >>= 7
    return output[::-1]

def _s2i(input, i=0, n=-1):
    """ helper for converting string to integer using ascii starting at position i, using n chars per entity """
    if n == -1:
        n = len(input)
    output = 0
    for j in range(i, min(i+n, len(input))):    
        output <<= 8
        char_id = ord(input[j])
        if char_id < 0 or char_id >= 256:
            char_id = ord(' ')
        output += char_id
    return output

def _i2s(input, i=0, n=-1):
    """ helper for converting int to string (inverse of _s2i) """
    output = ''
    while input:
        output += chr(input & 0xff)
        input >>= 8
    return output[::-1]

def _bitarr2bytes(input, pad=None, padsz=8):
    """ helper for converting bit array to bytes """
    output = []
    if pad:
        input = ([0] * padsz) + input
        if len(input) % pad:
            padlen = pad - len(input) % pad
            input.extend(np.random.randint(0, 2, padlen)) # add some garbage
            padlen_mark = _str2bitarr(_i2s(padlen % 256))
            if len(padlen_mark) < 8:
                padlen_mark = [0] * (8 - len(padlen_mark)) + padlen_mark
            if padsz == 16:
                padlen_mark = _str2bitarr(_i2s(padlen // 256)) + padlen_mark
                if len(padlen_mark) < 16:
                    padlen_mark = [0] * (16 - len(padlen_mark)) + padlen_mark
            input[0:padsz] = padlen_mark
    for i in range(0, len(input), 8):
        val = 0
        for j in range(i, min(i+8, len(input))):
            val <<= 1
            val += input[j]
        output.append(val)
    return bytes(output)

def _str2bitarr(input):
    """ helper for converting string to bit array """
    output = []
    input = [ord(x) for x in input]
    for i in range(0, len(input)):
        x = 0x80
        for j in range(8):
            output.append(1 if (input[i] & x) else 0)
            x >>= 1
    return output

def _bytes2bitarr(input):
    """ helper for converting bytes to bit array """
    output = []
    input = [x for x in input]
    for i in range(0, len(input)):
        x = 0x80
        for j in range(8):
            output.append(1 if (input[i] & x) else 0)
            x >>= 1
    return output

def _unpad(input, padsz = 8):
    """ remove padding from the end of a bit array, assuming first two bytes are padding size """
    if len(input) < padsz:
        return input
    padlen = int(_bitarr2bytes(input[:8], False)[0])
    if padsz == 16:
        padlen = padlen * 256 + int(_bitarr2bytes(input[8:16], False)[0])
    if padlen:
        return input[padsz:-padlen]
    else:
        return input[padsz:]

class HuffDict:
    """ Huffman code dictionary, based on Trie """

    @staticmethod
    def from_text(data, n):
        """ create binary huffman code dictionary from the given text with n characters per entity """
        from heapq import heappush, heappop, heapify
        num_keys = 128 ** n
        freq_dict = [0] * num_keys

        for i in range(0, len(data), n):
            freq_dict[_s2i_128(data, i, n)] += 1

        heap = [HuffDict(freq_dict[x], val=x) for x in range(num_keys)]
        heapify(heap)
        while len(heap) > 1:
            node_x = heappop(heap)
            node_y = heappop(heap)
            heappush(heap, node_x + node_y)
        heap[0]._len = num_keys
        heap[0]._n = n
        return heap[0]

    @staticmethod
    def from_text_file(path, n):
        """ create binary huffman code dictionary from text file at the given path with n characters per entity """
        f = open(path, 'r', encoding='utf-8')
        return HuffDict.from_text(f.read(), n)

    @staticmethod
    def from_save(path):
        """ load saved HuffDict from path """
        f = open(path, 'rb')
        import pickle
        return pickle.load(f)

    def __init__(self, freq=0., val=None, child0=None, child1=None, parent=None):
        self.freq = freq
        self.parent = parent
        self.child0 = child0
        if child0:
            child0.parent = self
        self.child1 = child1
        if child1:
            child1.parent = self
        self.val = val

    def __lt__(self, other):
        """ for heap comparison """
        return self.freq < other.freq

    def __add__(self, other): 
        """ merge nodes """
        return HuffDict(self.freq + other.freq, child0 = self, child1 = other)

    def reverse_dict(self): 
        """ construct reverse-lookup dict """
        if not hasattr(self, '_rdict'):
            self._rdict = {}
            for code, val in self:
                self._rdict[val] = code
        return self._rdict

    def save(self, path):
        """ save to path """
        import pickle
        self.reverse_dict()
        f = open(path, 'wb')
        pickle.dump(self, f)

    def __repr__(self):
        if not hasattr(self, '_repr'):
            s = ['HuffDict {\n']
            for code, val in self:
                s.append('   {}:{}\n'.format(''.join(map(str,code)), val))
            s.append('}')
            self._repr = ''.join(s)
        return self._repr

    def __getitem__(self, key):
        """ encode string => code, OR decode code (list/tuple/int) => string """
        if type(key) is str:
            return self.encode(key)
        elif type(key) is list or type(key) is tuple:
            return self.decode(key)
        else: # int: one at a time only
            node = self
            while node:
                if node.val and not key:
                    return _i2s_128(node.val)
                if key & 1:
                    node = node.child1
                else:
                    node = node.child0
                key >>= 1
            return None

    def __len__(self):
        if hasattr(self, '_len'):
            return self._len
        else:
            return 1

    def __iter__(self):
        node, pnode = self, None
        code = []
        while True:
            tmp = node
            if node.val is not None:
                yield (code[:], node.val)
            if (pnode is None or pnode is node.parent) and node.child0:
                node = node.child0 
                code.append(0)
            elif (pnode is node.child0 or \
                    ((pnode is None or pnode is node.parent) and not node.child0)) \
                     and node.child1:
                node = node.child1
                code.append(1)
            else: 
                node = node.parent
                if not code:
                    raise StopIteration
                code.pop()
            pnode = tmp
    
    def encode(self, input):
        """ encode an input string with this encoder """
        n = self._n
        result = []
        if len(input) % n:
            input += ' ' * (n - len(input) % n)
        for i in range(0, len(input), n):
            i = _s2i_128(input, i, n)
            result.extend(self.reverse_dict()[i])
        return result

    def decode(self, seq):
        """ decode an encoded bit sequence (list) with this encoder """
        output = []
        pos = 0
        while pos < len(seq):
            node = self
            new_pos = pos
            for i in range(pos, len(seq)):
                if not node or node.val:
                    break
                if seq[i] & 1:
                    node = node.child1
                else:
                    node = node.child0
                new_pos = i+1
            if node.val:
                output.append(_i2s_128(node.val))
                pos = new_pos
            else:
                pos += 1
        return ''.join(output)
    
