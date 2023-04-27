library(tuneR)
library(warbleR)

# This first block shows how to query and download specic species vocalizations from xeno-canto. 
# Then, it shows the conversion from .mp3 tp .wav

# Load rail calls from xeno-canto, set download = TRUE to download them to 'path'. When set to false,
# this returns as a data.frame important metadata about the files and it should be saved.
rallus <- query_xc(qword = "Rallus limicola", download = FALSE, path = 'rail_sounds')


# Set the path to the directory containing the MP3 files
mp3_dir <- "rail_sounds"

# Set the path to the directory where the WAV files will be saved
wav_dir <- "rail_wavs"

# Get a list of all the MP3 files in the directory
mp3_files <- list.files(mp3_dir, pattern = ".mp3", full.names = TRUE)

# Loop through each MP3 file and save a WAV file with the same name
for (file in mp3_files) {
  mp3 <- readMP3(file)
  wav_file <- gsub("\\.mp3", ".wav", basename(file))
  writeWave(mp3, file.path(wav_dir, wav_file))
}



# List all .wav files in directory
wav_files <- list.files(wav_dir, pattern = 'wav', full.names = TRUE)

# Create a data.frame to store file metadata
file_info <- data.frame(filename = NA, sampling_rate = NA, stereo = NA)

# Loop through files to extract sampling rate and number of channels
for (i in 1:length(wav_files)) {
  wav <- readWave(wav_files[i])
  rate <- wav@samp.rate
  if(!class(wav) == "Wave")
  {channels <- "multichannel"
  }else{
    channels <- wav@stereo
  }
  
  file_info[i, 'filename'] <- wav_files[i]
  file_info[i, 'sampling_rate'] <- rate
  file_info[i, 'stereo'] <- channels
}


# Stereo to mono conversion
for (file in wav_files){
  # Read the wav file
  temp_wav <- readWave(file)
  
  # If stereo, apply mono() function
  if(temp_wav@stereo) {
    temp_wav <- mono(temp_wav, which = 'both')
  }
  # Write new wav file to same location
  writeWave(temp_wav, filename = file)
}



# Downsample to 44100 -- Note: We chose 44100 based on the data we saw in file_info. Most wav files were 48000 or 44100,
# and you cannot 'upsample'. However, if there are any files that are lower than 44100 they will not be resampled and 
# should be discarded. For us, this was only a few files.

for (file in wav_files) {
  temp_wav <- readWave(file)
  if(!temp_wav@samp.rate == 44100){
    temp_wav <- downsample(temp_wav, 44100)
    writeWave(temp_wav, file)
  }
}

dir.create('discard_folder')

to_discard <- file_info[file_info$sampling_rate < 44100, "filename"]

file.rename(from = to_discard, to = paste('discard_folder', basename(to_discard), sep = '/'))


# Use file_info loop to check once more and ensure common sampling rate and number of channels
                                 