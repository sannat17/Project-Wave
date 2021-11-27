import librosa
import librosa.display
import soundfile
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft2
from scipy.fft import rfft, rfftfreq, irfft


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
dft_y = rfft(y)
dft_x = rfftfreq(num_samples, 1 / sampling_rate) # Gives the bins


## Apply inverse DFT to regain original wave

### I am removing all frequencies that have amplitude less than <min_amplitude_ratio> times max amplitude
### NOTE: There are better ways to extract the most useful features.. this is still naive
min_amplitude_ratio = 0.1 # w.r.t. the max amplitude
min_amplitude_req = np.max(np.abs(dft_y)) * min_amplitude_ratio
mask = np.abs(dft_y) < min_amplitude_req
dft_y_augmented = dft_y.copy()
dft_y_augmented[mask] = 0

reproduced_sig = irfft(dft_y_augmented)
    # Save reproducted signal in a wav file just to compare if it sounds similar
soundfile.write("reproduced_example_StarWars.wav", reproduced_sig, sampling_rate)
librosa.display.waveshow(reproduced_sig, sr=sampling_rate, ax=ax[1])
ax[1].set(title="Audio converted back to time domain by inverse DFT")
ax[1].label_outer()
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")


# Save plot of all stages
fig.tight_layout()
plt.savefig("example.png")

plt.close()

# Plotting the input audio's frequency domain of only top 10 frequencies
mask = np.abs(dft_y_augmented) > 0
last_nonzero = len(mask) - np.flip(mask).argmax() - 1

plt.plot(dft_x[:last_nonzero+10], np.abs(dft_y_augmented)[:last_nonzero+10])

plt.title(f"Top {np.count_nonzero(mask)} peaks of input audio's frequency domain")
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.savefig("example_frequency_domain.png")