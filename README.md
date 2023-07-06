# RailNET, a deep learning model for acoustic monitoring of elusive species like the Virginia Rail

With biodiversity in rapid decline worldwide, efficient wildlife monitoring solutions are increasingly important. Advances in deep learning have led to new opportunities in acoustic monitoring, a field where fully manual annotation of audio recordings in prohibitively labor-intensive. State-of-the-art models for bird monitoring perform well in general, but performance varies with rare, elusive species and generalization from the focal recordings in training data to soundscape data collected in monitoring programs has proved to be a major challenge. This project focuses on the Virginia rail--a cryptic marsh bird that can serve as a stand-in for other elusive monitoring targets for whom species-specific classifiers might be of interest.

While I hope RailNET might provide some value as a classifier for Virginia rail vocalizations, my larger hope is that the methodology used to develop RailNET can provide guidance to any project seeking to use deep learning audio classification models to monitor rare, elusive targets where high recall is of the utmost importance. I developed RailNET because I happened to be studying cryptic marsh birds and wanted a more reliable solution, but I am also interested in using similar methodology to support monitoring of other elsuive species, avian and otherwise. 

## Getting Started
A bulk of this repo is made up of code that was used during experimentation and model training. If you are interested in these details, exploring the src folder in conjunction with the technical report would be a good place to start. 

If you're primarily interested in using RailNET, then the inference_example.py will load in the model and run sample inferences on two sample audio files in the data folder, and this can be modified to point to your own files.

The data curation and labeling scripts are in the .R files. This includes the general workflow I used for pulling data from Xeno-canto and the interactive annotation function that makes applying strong labels efficient and easy. I hope to soon make the data curation scripts more modular so that anyone can easily download and preprocess data from Xeno-canto and feed it directly into a model training pipeline. 

### Installation

Necessary dependencies are located in the requirements.txt file.

```
pip install -r requirements.txt
```
## Usage

To run an inference demo on two sample files in the 'data//sample_audio' folder, in the command terminal, run:

```
py inference_example.py
```
After a bit of runtime (a progress bar should appear), this will produce a .csv file in your sample_audio folder showing showing prediction for each three-second segment of the orginal audio files, with confidence scores for each constituent model (RailNET is an ensemble), as 
well as the final ensemble prediction. For reference, the audio file called "many_calls" has 300+ rail vocalizations and the "couple_grunts" file has two.

To use RailNET on your own recordings, simply modify the code in this file to point to your chosen input folder and output folder. When creating the instance of EnsembleInference(), you can adjust the threshold cutoff and whether or not you want detections for all three-second segments or just the ones that qualify as detections. 

```
inference_call = EnsembleInference(RailNET, preprocessing_transforms, threshold=0.5)
```

The clean_output is True by default and the threshold is 0.5.

Keep in mind that the negative class that this model was trained on reflects the specific needs of my study area--meaning that precision is likely to suffer if your background soundscapes differ from the ones that informed our decisions of what species to include. Recomposing the negative class and retraining could help here, and I hope to develop some processes to streamline this type of recomposition in the near future. For now, you can use the train_RailNET.py file to find code to train an instance of RailNET, provided you have folders for a positive and negative class. The usage example at the bottom of the script shows how you can do this:

```
 train_RailNET(mod_type= 'resnet34',  
               augmentation_level= 'moderate', 
               positive_train= "..//data//positive_class", 
               negative_train= "..//data//negative_class",
               positive_test='..//data//positive_soundscape',
               negative_test='..//data//negative_soundscape',
               num_epochs=70,
               starting_LR=0.001,
               checkpoint_path="..//..//name_for_checkpoint.pth")
```

Note that for this to be effective, it is best to use true soundscape data for the test case. You could also use a subset of focal recordings from public databases, but high performance on these will not always be reflected in high performance on soundscape data. Since the train function used here saves checkpoints with the best recall and the best F1 score on the test set, using real monitoring data from soundscape recordings will lead to the best performance.

I will be using RailNET in monitoring projects for the next few months, hoping to further test its ability to generalize. In addition to further testing, I hope to expand its capacity to include more cryptic marsh birds, including black rails, yellow rails, king rails, American bittern, and least bittern, ultimately shifting to a multiclass framework that should prove to be more flexible than the binary classifier it is currently set up as.

## Collaborating

I freelance as a consultant for acoustic monitoring projects, and I would love to help you with yours!
## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You are free to use, share, and adapt this work for non-commercial purposes only, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. For more information, please visit [creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).

