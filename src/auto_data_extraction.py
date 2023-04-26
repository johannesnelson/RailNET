#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import librosa
import numpy as np
import soundfile as sf


def split_audio(input_folder, output_folder, target_sr=44100, window_seconds=3, threshold_multiple=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all the files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            filepath = os.path.join(input_folder, file)
            y, sr = librosa.load(filepath, sr=target_sr)

            windows, energies = calculate_energies(y, sr, window_seconds)
            signal_windows = identify_signal_windows(energies, threshold_multiple)
            signal_samples = signal_windows * windows.shape[1]

            save_segments(y, signal_samples, file, output_folder, sr, window_seconds)

    print("Processing complete. Check the '{}' folder for the {}-second segments.".format(output_folder, window_seconds))


def calculate_energies(y, sr, window_seconds):
    window_length = window_seconds * sr
    hop_length = window_length
    windows = librosa.util.frame(y, frame_length=window_length, hop_length=hop_length).T
    energies = np.sum(windows ** 2, axis=1)
    return windows, energies


def identify_signal_windows(energies, threshold_multiple):
    energy_threshold = (threshold_multiple * energies.mean())
    signal_windows = np.where(energies > energy_threshold)[0]
    return signal_windows


def save_segments(y, signal_samples, file, output_folder, sr, window_seconds):
    window_length = window_seconds * sr
    for i, start_sample in enumerate(signal_samples):
        end_sample = start_sample + window_length
        output_filename = f"{os.path.splitext(file)[0]}_segment_{i}.wav"
        output_filepath = os.path.join(output_folder, output_filename)
        sf.write(output_filepath, y[start_sample:end_sample], sr)


# # Usage example
# input_folder = "rail_wavs"
# output_folder = "auto_training_data"
# target_sr = 44100  # Desired sampling rate
# threshold_multiple = 1.5

# split_audio(input_folder, output_folder, target_sr=target_sr, threshold_multiple=threshold_multiple)

