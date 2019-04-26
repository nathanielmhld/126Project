# 126 Project - Audio Transmission

- Run `transmit.py` to transmit
  - `python transmit.py` transmits sample text
  - `python transmit.py "text"` transmits specified text
  - `python transmit.py -f file.txt` transmits a text file

- Run `receive.py` to start up a receiver.
  - if debug mode is enabled in config, then use `receive.py -f` to re-decode last message for quiet testing

- See config options in `config.py`, for ex. enabling debug mode, changing frequencies used, etc.
