import argparse
import os
import shutil

import pandas as pd

from pathlib import Path
from tqdm import tqdm

DATA_CSV_PATH = "./data/vggsound.csv"

def create_subset(keywords:str = "playing volleyball", max_size:int = 500, include_test:bool = False):
    Path("./data/subset/audio").mkdir(parents=True, exist_ok=True)
    Path("./data/subset/video").mkdir(parents=True, exist_ok=True)

    if not os.path.exists(DATA_CSV_PATH):
        return 
    df = pd.read_csv(DATA_CSV_PATH, header = None)

    if not include_test:
        df = df[df.iloc[:, 3]=="train"]
    
    df = df[df.iloc[:, 2]==keywords] # filter by captions

    data_folder_path = Path("./data/") 

    df["file_name"] = df.iloc[:,0] + "_" + df.iloc[:,1].apply(lambda x: str(1000000+x)[1:]) + ".mp4"
    df["video_path"] = df["file_name"].apply(lambda x: data_folder_path / "video" / x)
    df["audio_path"] = df["file_name"].apply(lambda x: (data_folder_path / "audio" / x).with_suffix(".wav"))
    df["target_video_path"] = df["file_name"].apply(lambda x: data_folder_path / "subset" / "video" / x)
    df["target_audio_path"] = df["file_name"].apply(lambda x: (data_folder_path / "subset" / "audio" / x).with_suffix(".wav"))

    df = df[df["video_path"].apply(lambda x: os.path.exists(x))] # filter out nonexist path
    df = df[df["audio_path"].apply(lambda x: os.path.exists(x))]
    temp_df = df.iloc[:max_size, :]       # trunck to target size
    valid_df = df.iloc[max_size:, :]       # trunck to target size
    valid_df.iloc[:, 3] = "test"

    final_df = pd.concat([temp_df, valid_df])

    total_moved = 0
    for i in tqdm(range(final_df.shape[0])):
        v_path, a_path, v_target, a_target = final_df.iloc[i, -4:]

        if os.path.exists(v_path):
            shutil.copy(v_path, v_target)
        if os.path.exists(a_path):
            shutil.copy(a_path, a_target)
            total_moved += 1
    print(f"{total_moved}")
    
    final_df = final_df.iloc[:, :4]
    final_df.columns = ["ytid","start_time","class","set"]
    final_df.to_csv(Path("./datasets") / "vggsound_subset.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, default="playing volleyball")
    parser.add_argument("--max_size", type=int, default=500)
    parser.add_argument("--include_test", type=bool, default=False)

    args = parser.parse_args()
    create_subset(args.keywords, args.max_size, args.include_test)

# In root folder: python3 -m utils.subset


