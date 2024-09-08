# the part that goes into its own file

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
from scipy.stats import entropy
import os
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)


class InceptionScoreCalculator:
    
    def __init__(self, vid_dir_path, device = "cpu", model = None): # there's definitely something or other that I've forgotten
        self.device = device
        
        if model is None:
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
            self.model = self.model.eval()
            self.model = self.model.to(self.device)
        else:
            self.model = model # assumes that self.model has been set to eval mode and sent to device already

        # specified for the model
        self.side_size = 256
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
        self.crop_size = 256
        self.num_frames = 8
        self.sampling_rate = 8
        self.frames_per_second = 30

        self.transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(
                        size=self.side_size
                    ),
                    CenterCropVideo(crop_size=(self.crop_size, self.crop_size))
                ]
            ),
        )

        self.clip_duration = (self.num_frames * self.sampling_rate)/self.frames_per_second

        self.start_sec = 0 # should correspond to where the action happens in the video, but set to 0 for this.
        self.end_sec = self.start_sec + self.clip_duration

        # get the videos
        self.vid_dir_path = vid_dir_path
        self.vid_names = [i for i in os.listdir(vid_dir_path) if (i[-4:] == '.mp4')]
        print(f"Inception Score: Using {len(self.vid_names)} videos in directory: {self.vid_dir_path}")

    def get_video_batch(self, batch_start = 0, batch_end = None):
        if batch_end is None:
            batch_end = len(self.vid_names)
        vid_data_tgt = []
        for vid_name in self.vid_names[batch_start:batch_end]:
          vid_file_path = self.vid_dir_path + vid_name
          vid_data = self.transform(EncodedVideo.from_path(vid_file_path).get_clip(start_sec=self.start_sec,end_sec=self.end_sec))["video"]
          vid_data_tgt.append(vid_data) # this is the bottleneck, not sure how to fix it.
        return(torch.stack(vid_data_tgt)) # end user must put it on the device
    
    # This one should be the one used
    def calculate_inception_score_batched(self, splits = 3): # splits default should be 10, not 3.
        scores = []
        for i in tqdm(range(splits)):
            start_idx = i * (len(self.vid_names) // splits)
            end_idx = (i + 1) * (len(self.vid_names) //
                                  splits) if i != splits - 1 else len(self.vid_names)
            batch = self.get_video_batch(batch_start = start_idx, batch_end = end_idx)
            batch = batch.to(self.device)
            with torch.no_grad():
                preds = self.model(batch).squeeze()
            prob_dist = F.softmax(preds.detach(), dim = 1).cpu()
            scores.append(entropy(prob_dist, qk = prob_dist.mean(dim = 0)))
        is_score = np.exp(np.mean(scores))
        return is_score
    
    # This is another option if we have a small total amount of videos to calculate over.
    def calculate_inception_score_unbatched(self, splits = 3): # splits default should be 10, not 3.
        batch = self.get_video_batch()
        batch = batch.to(self.device)
        return self.calculate_inception_score_og(batch, splits)
    
    def calculate_inception_score_og(self, generated_vids, splits):
        # this also assumes that generated_vids is on the device already. 
        with torch.no_grad():
            preds = self.model(generated_vids).squeeze() # uses resnet3d as the model for this - i guess this part has to be batched.

        scores = []
        for i in tqdm(range(splits)):
            start_idx = i * (len(preds) // splits)
            end_idx = (i + 1) * (len(preds) //
                                  splits) if i != splits - 1 else len(preds)
            prob_dist = F.softmax(preds.detach()[start_idx:end_idx], dim=1).cpu()
            scores.append(entropy(prob_dist, qk = prob_dist.mean(dim=0))) # make sure it's calculating the kl divergence

        is_score = np.exp(np.mean(scores)) # this seems a little weird. should recalculate the inception score stuff. check this!

        return is_score