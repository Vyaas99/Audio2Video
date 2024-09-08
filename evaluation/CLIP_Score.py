# this goes in its own class.

import os
import pandas as pd
import clip
from PIL import Image
import cv2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class CLIPScoreCalculator:
    def __init__(self, vid_dir_path, text_desc = None, model = None, device="cuda", csv_path = None): # change the csv path as necessary
        # csv path is where the csv to the corresponding text inputs are stored
        self.device = device
        if model is None:
            self.model, _ = clip.load("ViT-B/32", device=device) # from clip import CLIP wasn't working
            self.model.eval()
        else:
            self.model = model # assume it's sent to device and is in eval mode already

        self.image_transform = Compose([
            Resize((224, 224)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # get video directory
        self.vid_dir_path = vid_dir_path
        self.vid_names = [i for i in os.listdir(vid_dir_path) if (i[-4:] == '.mp4')]
        print(f"CLIP Score: Using {len(self.vid_names)} videos in directory: {self.vid_dir_path}")
        
        # if text desc is provided this is used (default):
        self.text_desc = text_desc

        # if text desc is not provided, use the csv file
        if not text_desc:
            self.text_descs = pd.read_csv(csv_path)
        else:
            self.text_descs = None
        

    def preprocess_image(self, image):
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
        return image_tensor

    # this is the final function to use
    def calculate_clip_score(self):
        total_vid_sim = 0
        count_vids = 0
        for vid_name in self.vid_names:
          vid_file_path = self.vid_dir_path + vid_name
          if self.text_desc:
              text_desc = self.text_desc
          else:
              temp = vid_name.split("_")
              vid_id = temp[0]
              vid_start_time = int(temp[1].split(".")[0])
              text_desc = list(self.text_descs[(self.text_descs['ytid'] == vid_id) & (self.text_descs['start_time'] == vid_start_time)]['class'])[0] # gett the text description form csv file
              if not text_desc:
                  print(f"Could not find text description for video id: {vid_id}, start time: {vid_start_time}")
                  break
          print("Using text description: " + text_desc)
          total_vid_sim += self.calculate_clip_score_video(vid_file_path, text_desc)
          count_vids += 1
        return total_vid_sim / count_vids

    def calculate_clip_score_video(self, video_path, text_description):
        text_input = clip.tokenize(
              [text_description]).to(self.device)
        text_features = self.model.encode_text(text_input)

        # loop over video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_sim = 0
        count_frames = 0
        while True:
            is_read, frame = cap.read()
            if not is_read:
              break
            frame = Image.fromarray(frame)
            image_tensor = self.preprocess_image(frame)
            image_features = self.model.encode_image(image_tensor)
            similarity_score = (
              torch.nn.functional.cosine_similarity(image_features, text_features)).item()
            total_sim += similarity_score
            count_frames += 1

        return total_sim / count_frames

    def calculate_clip_score_image(self, image, text_description): # changing this to take an image instead of image path
        image_tensor = self.preprocess_image(image)
        text_input = clip.tokenize(
            [text_description]).to(self.device)

        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_input)

        similarity_score = (
            torch.nn.functional.cosine_similarity(image_features, text_features)).item()

        return similarity_score