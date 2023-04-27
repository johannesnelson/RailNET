# RailNET, a deep learning model for detecting Virginia rail vocalizations

With biodiversity in rapid decline worldwide, efficient wildlife monitoring solutions are increasingly important. Deep learning models offer new opportunities in acoustic monitoring, a field where fully manual annotation of audio recordings in prohibitively labor-intensive. State-of-the-art models for bird monitoring perform well in general, but performance varies with rare, elusive species and generalization from the focal recordings in training data to soundscape data collected in monitoring programs has proved to be a major challenge. This project focuses on the Virginia rail--a cryptic marsh bird that can serve as a stand-in for other elusive monitoring targets for whom species-specific classifiers might be of interest.

So, while I hope RailNET might provide some value as a classifier for Virginia rail vocalizations, my larger hope is that the methodology I used to develop RailNET can provide guidance to any project seeking to use deep learning audio classification models to monitor rare, elusive targets. I developed RailNET because I happened to be studying cryptic marsh birds, but I am also interested in using similary methodology to support monitoring of various elsuive species, avian and otherwise. 
## Getting Started

A bulk of this repo is made up of code that was used during experimentation and model training. If you are interested in these details, exploring the src folder in conjunction with the technical report would be a good place to start.

If you're primarily interested in using RailNET, then the inference_example.py will load in the model and run sample inferences on two sample audio files in the data folder, and this can be modified to point to your own files.

The data curation and labeling scripts are in the .R files, and are not runnable as they are. Since this part of the process involves a lot of manual intervention, these are more interactive functions. I hope to soon make the data curation scripts more modular so that anyone can easily downlaod and preprocess data from Xeno-canto.

### Installation

Necessary dependencies are located in the requirements.txt file, and can be installed with the following terminal command, keeping in mind that the .txt file needs to be in your current working directory:

```
pip install -r requirements.txt
```
## Usage

To run an inference demo on two sample files in the 'data//sample_audio' folder, run:

```
py inference_example.py
```
After a bit of runtime, this will produce a .csv file in your sample_audio folder showing showing prediction for each three-second segment of the orginal audio files, with confidence scores for each constituent model (RailNET is an ensemble), as 
well as the final ensemble prediction. For reference, the audio file called "many_calls" has 300+ rail vocalizations and the "couple_grunts" file has two.

To use RailNET on your own recordings, simply modify the code in this file to point to your chosen input folder and output folder. Keep in mind that the negative class that this model was trained on reflects the specific needs of my study area--meaning that precision is likely to suffer if your background soundscapes differ from the ones that informed my decision of what species to include. Recomposing the negative class and retraining could help here, and I hope to develop some processes to streamline this type of recomposition in the near future, which I will share.

I will be using RailNET in monitoring projects for the next few months, hoping to further test its ability to generalize. In addition to further testing, I intend to expand its capacity to include more cryptic marsh birds, including black rails, yellow rails, king rails, American bittern, and least bittern, though wheter these tests are conclusive or not will depend on the pretty-unlikely event that many of these birds are even around in the first place.

## Collaborating

If you have an acoustic monitoring project and want to develop deep learning models to support your efforts, don't hesitate to reach out!

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. You are free to use, share, and adapt this work for non-commercial purposes only, as long as you give appropriate credit, provide a link to the license, and indicate if changes were made. For more information, please visit [creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).

