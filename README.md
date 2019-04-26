# 126 Project - Audio Transmission

- To begin, please download my Huffman code model for English, built from War and Peace: [https://www.ocf.berkeley.edu/~sxyu/huffman_model.pkl](https://www.ocf.berkeley.edu/~sxyu/huffman_model.pkl)
  - Place it in the project root directory, that is, alongside `receive.py`
  - Note: this is an updated model more efficient than before

- Run `transmit.py` to transmit
  - `python transmit.py` transmits sample text
  - `python transmit.py "text"` transmits specified text
  - `python transmit.py -f file.txt` transmits a text file

- Run `receive.py` to start up a receiver.
  - if debug mode is enabled in config, then use `receive.py -f` to re-decode last message for quiet testing

- See config options in `config.py`, for ex. enabling debug mode, changing frequencies used, etc.
