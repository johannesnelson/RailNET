#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.metrics import roc_curve, auc



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
    plt.title(f'{title} (Label: {label})') 
    plt.colorbar(format='%+2.0f dB')
    plt.show()


# In[ ]:


def plot_roc_curve(fpr, tpr, roc_auc, all_targets, all_probabilities, title, desired_thresholds=None, filename=None):
    num_classes = len(fpr)
    cmap = plt.get_cmap('coolwarm')

    fpr_i, tpr_i = fpr[1], tpr[1]
    roc_auc_i = roc_auc[1]

    fpr_values, tpr_values, thresholds = roc_curve(all_targets, [prob[1] for prob in all_probabilities])

    confidence_levels = np.interp(thresholds, (0, 1), (0, 1))

    if desired_thresholds is None:
        desired_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    closest_threshold_indices = [np.argmin(np.abs(thresholds - t)) for t in desired_thresholds]

    for j in range(len(fpr_i) - 1):
        plt.plot(fpr_i[j:j + 2], tpr_i[j:j + 2], color=cmap(confidence_levels[j]), lw=2, zorder=1)

        if j in closest_threshold_indices:
            marker_label = f"{thresholds[j]:.1f}"
            plt.scatter(fpr_i[j], tpr_i[j], marker="o", label=marker_label, edgecolors="k", color=cmap(confidence_levels[j]), zorder=2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{title}\nAUC: {roc_auc_i:.4f}", fontsize=10)


    # Add legend for threshold markers
    plt.legend(title="Thresholds", loc="lower right")

    sm = ScalarMappable(cmap=cmap)
    sm.set_array([])  
    cbar = plt.colorbar(sm)
    cbar.set_label('Threshold Cutoffs')

    if filename:
        plt.savefig(filename)
    plt.show()

    

