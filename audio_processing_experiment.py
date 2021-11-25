import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft


# Trying to recreate the wav file using DFT and then inverse DFT
audio_file = "./example_StarWars_3sec.wav"
y, sampling_rate = librosa.load(audio_file)
num_samples = len(y)
duration = num_samples / sampling_rate

# Set plt subplots to plot intermediate results
fig, ax = plt.subplots(2, 1)

## Input audio waveform before DFT
librosa.display.waveshow(y, sr=sampling_rate, ax=ax[0])
ax[0].set(title=f'Input audio (sampling rate: {sampling_rate})')
ax[0].label_outer()
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amplitude")

## Apply DFT to convert wave to frequency domain
dft_y = fft(y)
dft_x = fftfreq(num_samples, 1 / sampling_rate) # Gives the bins


## Apply inverse DFT to regain original wave
### I am picking 10 peaks to remove noise
reproduced_sig = np.abs(ifft(dft_y))
librosa.display.waveshow(reproduced_sig, sr=sampling_rate, ax=ax[1])
ax[1].set(title="Audio converted back to time domain by inverse DFT")
ax[1].label_outer()
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")

# Save plot of all stages
fig.tight_layout()
plt.savefig("example.png")

plt.close()

# Plotting the input audio's frequency domain
plt.plot(dft_x, np.abs(dft_y))
plt.xlim(left=0)
plt.title("Input audio in frequency domain")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("example_frequency_domain.png")