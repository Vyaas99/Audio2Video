#!/bin/zsh

#hopefully this will convert the mp4s into wav files

for i in $(ls *.wav); do
filename="${i%.*}"
ffmpeg -i $filename.wav -ar 16000 resampled/$filename.wav
done

# and then will manually move the audios for now.
