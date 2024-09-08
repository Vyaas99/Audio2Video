#!/bin/zsh

#hopefully this will convert the mp4s into wav files

for i in $(ls *.mp4); do
filename="${i%.*}"
ffmpeg -i $filename.mp4 $filename.wav
done

# and then will manually move the audios for now.
