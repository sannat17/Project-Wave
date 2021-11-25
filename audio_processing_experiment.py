import librosa
import librosa.display
import matplotlib.pyplot as plt


# Trying to recreate the wav file using DFT and then inverse DFT
audio_file = "./example_StarWars_3sec.wav"
y, sr = librosa.load(audio_file)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

## Input audio waveform before DFT
librosa.display.waveplot(y, sr=sr, ax=ax[0])
ax[0].set(title=f'Input audio (sampling rate: {sr}')
ax[0].label_outer()
librosa.display.waveplot(y, sr=sr)


plt.savefig("example.png")