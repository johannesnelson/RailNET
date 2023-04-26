#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
import torchaudio
import numpy as np
import pandas as pd


class EnsembleInference:
    def __init__(self, models, preprocessing_transforms, device=None, threshold=0.5):
        self.models = models
        self.transforms = preprocessing_transforms
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        for model in self.models:
            model.to(self.device)

    def predict(self, filepath):
        audio, sr = torchaudio.load(filepath)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100)
        audio = resampler(audio)
        sr = 44100

        segment_size = 3
        hop_size = 3
        n_segments = int(np.floor((audio.shape[1] - segment_size * sr) / (sr * hop_size))) + 1

        predictions = []

        for i in range(n_segments):
            start_time = i * hop_size
            end_time = start_time + segment_size

            if end_time > audio.shape[1] / sr:
                continue

            segment = audio[:, int(start_time * sr):int(end_time * sr)]

            segment = self.transforms(segment)
            segment = segment.unsqueeze(0)

            segment = segment.to(self.device)

            ensemble_confidence_scores = []

            for model in self.models:
                model.eval()
                with torch.no_grad():
                    prediction = model(segment)

                prediction = torch.nn.functional.softmax(prediction, dim=1)
                ensemble_confidence_scores.append(prediction[0, 1].item())

            avg_confidence_score = np.mean(ensemble_confidence_scores)
            ensemble_label = int(avg_confidence_score > self.threshold)

            model_predictions = {'filepath': filepath, 'start': start_time, 'end': end_time}
            for idx, score in enumerate(ensemble_confidence_scores):
                model_predictions[f'model_{idx}_confidence'] = score
                model_predictions[f'model_{idx}_prediction'] = int(score > self.threshold)

            model_predictions['ensemble_prediction'] = ensemble_label
            model_predictions['ensemble_confidence'] = avg_confidence_score

            predictions.append(model_predictions)

        df = pd.DataFrame(predictions)

        return df

    def process_folder(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        num_files = len(os.listdir(input_folder))
        print(f"Found {num_files} files in input folder...")


        for filename in os.listdir(input_folder):
            if filename.endswith(".wav") or filename.endswith(".WAV"):
                input_filepath = os.path.join(input_folder, filename)
                output_filename_no_ext, ext = os.path.splitext(filename)
                output_filename = output_filename_no_ext + '_results' + '.csv'
                output_filepath = os.path.join(output_folder, output_filename)

                print(f"Processing {input_filepath}...")
                df = self.predict(input_filepath)
                df.to_csv(output_filepath, index=False)
                print(f"Processed {input_filepath} and saved results to {output_filepath}")

