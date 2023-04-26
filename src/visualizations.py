#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt


# In[ ]:


def plot_random_spectrogram(dataset, title):
    random_index = np.random.randint(len(dataset))
    audio, label = dataset[random_index]
    print(audio.shape)
    audio = audio.squeeze().numpy()

    plt.figure()
    librosa.display.specshow(audio, sr=44100)
    plt.xlabel('Time')
    plt.ylabel('Frequency (mel)')
    plt.title(f'{title} (Label: {label})')  # Include the label in the title
    plt.colorbar(format='%+2.0f dB')
    plt.show()

