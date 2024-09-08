import multiprocessing
import os
import pandas as pd
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm

def multiprocessing_func(function, splitable_tasks, num_processes:int = 4):
    processes = []

    # Create and start each process
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker, args=(function, splitable_tasks[i::num_processes]))
        processes.append(process)
        process.start()
    
    # Wait for all processes to finish
    for process in processes:
        process.join()

    print("All processes have finished.")

def worker(function, tasks):
    for task in tqdm(tasks):
        function(task)

def extract_audios(dataset_file_path):
    if os.path.exists(dataset_file_path):
        df = pd.read_csv(dataset_file_path,header = None)
        df["file_path"] = "./data/video/" + df.iloc[:,0] + "_" + df.iloc[:,1].apply(lambda x: str(1000000+x)[1:]) + ".mp4"

    multiprocessing_func(extract_audio, df["file_path"], 100)

def extract_audio(video_path, audio_path = None, save = True):
    if not os.path.exists(video_path):
        return 
    if audio_path is None:
        vp = Path(video_path)
        audio_path = vp.parent.parent / "audio" / vp.name
        audio_path = audio_path.with_suffix(".wav")

    # Load the video clip
    video_clip = VideoFileClip(video_path)

    # Extract the audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio to a separate file
    if save and not os.path.exists(audio_path):
        audio_clip.write_audiofile(audio_path)

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    return str(audio_path)

if __name__ == "__main__":
    csv_path = "./data/vggsound.csv"

    extract_audios(csv_path)

# Linux code:
#   python3 -m utils.extract_audio