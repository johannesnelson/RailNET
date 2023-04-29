
import random
from pathlib import Path
from shutil import copy2

'''
This is only helpful if you have a number of folders with preprocessed negative class species samples, which can be created using
the .R scripts to query from xeno canto. I am testing out if recomposing the negative class for different survey areas helps maintain
precision. If it does, I will make the xeno-canto workflow more modular and easy to execute.

The noise_folder flag is something I am testing in composing a noise folder for the augmentation_noise_transforms, since currently I am
just using noise from random recordings I own.

'''


def create_negative_class(master_folder: str, total_samples: int, create_noise_dir: bool = False):
    master_path = Path(master_folder)
    subfolders = [f for f in master_path.iterdir() if f.is_dir()]

    samples_per_folder = total_samples // len(subfolders)
    negative_class_path = master_path / "negative_class"
    negative_class_path.mkdir(exist_ok=True)
    
    if create_noise_dir:
        noise_folder_path = master_path / "noise_folder"
        noise_folder_path.mkdir(exist_ok=True)

    for folder in subfolders:
        wav_files = list(folder.glob("*.wav"))
        random.shuffle(wav_files)
        selected_files = wav_files[:samples_per_folder]
        noise_files = wav_files[samples_per_folder:] if create_noise_dir else []

        for file in selected_files:
            copy2(file, negative_class_path)
        
        if create_noise_dir:
            for file in noise_files:
                copy2(file, noise_folder_path)