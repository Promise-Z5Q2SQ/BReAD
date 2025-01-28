import os
import torch
from tqdm import tqdm
from PIL import Image


def clip_features_cal(image_dataset_dir, feature_extractor, image_encoder):
    if os.path.exists(os.path.join(image_dataset_dir, "clip_features.pth")):
        return torch.load(os.path.join(image_dataset_dir, "clip_features.pth"))
    else:
        device = image_encoder.device

        image_files = []
        clip_dict = {}
        for root, dirs, files in os.walk(image_dataset_dir):
            for file in files:
                if file.upper().endswith((".JPEG")):
                    image_files.append(file)
        with torch.no_grad():
            print("----- Calculating CLIP embeddings -----", flush=True)
            for image_file in tqdm(image_files):
                image = Image.open(os.path.join(image_dataset_dir, image_file.split("_")[0], image_file))
                processed = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
                features = image_encoder(processed).image_embeds
                clip_dict[image_file] = features.cpu()
        torch.save(clip_dict, os.path.join(image_dataset_dir, "clip_features.pth"))
