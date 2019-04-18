import pyaudio
import numpy as np

p = pyaudio.PyAudio()

volume = 0.2     # range [0.0, 1.0]
fs = 44000       # sampling rate, Hz, must be integer
duration = 10   # in seconds, may be float
#f = 1000.0        # sine frequency, Hz, may be float
message = [440,880,1000]
# generate samples, note conversion to float32 array
samples = lambda f:(np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play. May repeat with different volume values (if done interactively)
for i in message: 
	#while True:			
  stream.write(volume*samples(i))
    #stream.write(0*samples(440))

stream.stop_stream()
stream.close()

p.terminate()