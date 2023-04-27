

library(tuneR)
library(data.table)



# This function allows users to annotate audio recordings for the presence of a specific species
# and save the results as a CSV file. It also includes options for playing the audio segment, 
# writing the segment as a WAV file, skipping segments, and adding notes. Additionally, the 
# function includes a "template mode" for creating specific time windows centered on detections 
# for use in machine learning training sets.

annotate_recording <- function(file_path, past_annos = NULL, page_length, temp_length = 3, species = NULL, sample_rate = 48000) {
  
  # Read the wave file and downsample if necessary
  wave_obj <- readWave(file_path)
  
  if (!wave_obj@samp.rate == sample_rate) {
    wave_obj <- downsample(wave_obj, sample_rate)
  }
  
  # Calculate wave duration and create sequence
  wave_dur <- length(wave_obj@left) / wave_obj@samp.rate
  wave_seq <- seq(0, wave_dur, page_length)
  
  # Create data table for annotations
  if (is.null(past_annos)) {
    annotationDT <- data.table(filepath = file_path, common_name = species, start = wave_seq, end = wave_seq + page_length, verification = NA, notes = NA)
  } else {
    annotationDT <- past_annos
  }
  
  # Set up options for user input
  verifs <- c()
  verif.options <- c("y", "n", "r", "q", "s")
  all.options <- c("y", "n", "r", "q", "p", "s", "w", "a", "t")
  
  
  template_DT <- data.table()
  
  
  # Iterate over each row in the data table
  for (i in 1:nrow(annotationDT)) {
    if (!is.na(annotationDT$verification[i])) {
      cat(paste("\n Verification for", basename(annotationDT$filepath[i]), "at", annotationDT$start[i], "seconds already exists. Moving onto next detection...\n"))
      annotationDT$verification[i] <- annotationDT$verification[i]
      next
    }
    
    # Main loop for displaying spectrograms and prompting user input
    repeat {
      
      # Display the spectrogram
      viewSpec(annotationDT$filepath[i], start.time = annotationDT$start[i], page.length = page_length, units = 'seconds')
      cat(paste("\n Showing detection", i, "out of", nrow(annotationDT), "from", basename(annotationDT$filepath[i]), "at", annotationDT$start[i], "seconds. Confidence:", annotationDT$confidence[i], "\n"))
      
      # Display input options
      cat(paste("Enter \n 'y' for yes,\n",  
                "'n' for no,\n",
                "'r' for review,\n",
                "'p' to play audio segment,\n", 
                "'w' to write segment as wav file to working directory,\n",
                "'s' to skip to next segment (and log as NA)",
                "'a' to add a note \n",
                "'q' for quit."))
      
      # Get user input
      answer <- readline(prompt = paste0(paste("Is the target species present?")))
      
      # If user input is valid, break the loop
      if (answer %in% verif.options) break
      
      # Play the audio segment
      if (answer == "p") {
        tempwave <- readWave(annotationDT$filepath[i], from = annotationDT$start[i], to = annotationDT$end[i], units = "seconds")
        play(tempwave)
      }
      # Write the audio segment to a wav file
      if (answer == "w") {
        filename <- paste0(paste(gsub(pattern = ".WAV", "", basename(annotationDT$filepath[i])), annotationDT$start[i], sep = "_"), ".WAV")
        tempwave <- readWave(annotationDT$filepath[i], from = annotationDT$start[i] - 1, to = annotationDT$end[i] + 1, units = "seconds")
        writeWave(tempwave, filename)
        cat("\n Writing wav file to working directory...")
      }
      
      # Add a note to the notes column
      if (answer == "a") {
        note <- readline(prompt = "Add note here: ")
        annotationDT$notes[i] <- note
      }
      
      # Template mode section (still in development)
      if (answer == "t") {
        
        repeat {
          wave.obj.2 <- readWave(annotationDT$filepath[i], from = annotationDT$start[i], to = annotationDT$end[i], units = 'seconds')
          tempSpec <- spectro(wave.obj.2, fastdisp = TRUE)
          t.bins <- tempSpec$time
          n.t.bins <- length(t.bins)
          which.t.bins <- 1:n.t.bins
          which.frq.bins <- which(tempSpec$freq >= 0)
          frq.bins <- tempSpec$freq
          amp <- round(tempSpec$amp[which.frq.bins, ], 2)
          n.frq.bins <- length(frq.bins)
          ref.matrix <- matrix(0, nrow = n.frq.bins, ncol = n.t.bins)
          
          if (temp_length == 'none') {
            t.value <- as.numeric(readline("How many seconds long would you like the templates to be?"))
          } else {
            t.value <- temp_length
          }
          
          cat("Click the plot where you would like to center this template")
          ctr.pt <- locator(n = 1)
          
          temp.DT <- data.table(filepath = annotationDT[i, filepath], 
                                common_name = annotationDT[i, common_name], 
                                start = (annotationDT[i, start]) + ctr.pt$x - (t.value/2), 
                                end = (annotationDT[i, start]) + ctr.pt$x +(t.value/2),
                                center.freq = ctr.pt$y)
          template_DT <-  rbind(template_DT, temp.DT)
          
          {break}
        }
        dev.off()
      }
      
      # If the answer is not recognized, inform the user
      if (!answer %in% all.options){
        cat("\n Response not recognized, please input correct response...\n")
      }
      
    }
    
    # Add user input to the verification column
    if (answer %in% c("y", "n", "r")) {
      cat("\n Adding result to verification data...\n ")
      annotationDT$verification[i] <- answer
    }
    # Skip the observation (leave as NA)
    if (answer == "s") {
      annotationDT$verification[i] <- NA
      cat("Skipping to next detection...")
    }
    
    # Quit the process and prompt user to save results
    if (answer == "q") {
      
      annotationDT$verification[i:nrow(annotationDT)] <- annotationDT$verification[i:nrow(annotationDT)]
      
      break
    }
    
  }
  # Prompt user to save results as a csv file
  
  # Ask user if they want to save results as a CSV file
  saveask <- readline(prompt = "Would you like to save results as a csv file? \n Input 'y' for yes:")
  if (saveask == "y") {
    fname <- readline(prompt = "What would you like to name the file?")
    
    write.csv(annotationDT, paste0(fname, ".csv"), row.names = FALSE)
  }
  
  # Ask user if they want to save the template data as a CSV file
  saveask2 <- readline(prompt = "Would you like to save the template data as a csv file? \n Input 'y' for yes:")
  if (saveask2 == 'y'){
    call_type <- readline(prompt = "What call type are these templates?")
    template_DT$vocType <- call_type
    csv_name <- readline(prompt = 'What would you like to name it?')
    write.csv(template_DT, paste0(csv_name, ".csv"), row.names = FALSE)
  }
  
  # Ask user if they want to return annotations or template data
  saveask3 <- readline(prompt = "Are you returning annotations or template data? \n Input 'a' for annotations or 't' for template data:")
  
  # Return the appropriate data based on the user's choice
  if (saveask3 == 'a') {
    return(annotationDT)
  }
  if (saveask3 == 't') {
    return(template_DT)
  }
  
} 



# After creating template annotations with the above script, this downloads the three-second chunks as individual .wav files
# Should be modified if not labeling by call type. Since most recordings from xeno-canto feature one call type within the 
# whole recording, we elected to track this to ensure balance with call types in final training dataset. 


makeWaves <- function(x, dir, time_buffer = 0, sampling_rate = 48000, vocType = "call") {
  dir.create(dir, recursive = TRUE)
  dir.fp <- dir
  for (i in 1:nrow(x)){
    wave <- readWave(x$filepath[i], from = x$start[i]-time_buffer, to = x$end[i]+time_buffer, units = 'seconds')
    if(!wave@samp.rate == sampling_rate){wave <- downsample(wave, sampling_rate)}
    
    writeWave(wave, filename = paste(dir.fp, paste(gsub(pattern = '.wav', x = basename(x$filepath[i]), replacement = ""), round(x$start[i], 2), paste0(vocType, ".wav"),  sep = '_'), sep = "/"))
    
  }
  
}
