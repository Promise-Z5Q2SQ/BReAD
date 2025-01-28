from embedding.model import get_eeg_encoder
from embedding.config import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
import torch
import os
from datetime import datetime
import json
from tqdm import tqdm
from embedding.splitter import Splitter
import gc
import numpy as np
import argparse
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from dataset import EEGImageNetDataset
from clip_features_cal import clip_features_cal
from de_feat_cal import de_feat_cal
from embedding.eeg_info_nce import *
from embedding.utils import retrievaler


def save_config(config, output_path):
    config_dict = config.__dict__
    config_file = os.path.join(output_path, "config.json")
    with open(config_file, "w") as f:
        json.dump(config_dict, f, indent=4)


def get_split_dataset(dataset, config, output_path):
    splitter = Splitter(config.train_ratio)
    train_dataset, test_dataset = splitter.split(dataset)
    splitter.save(os.path.join(output_path, "splitter.json"))
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument(
        "-g", "--granularity", required=True, help="granularity chosen from coarse, fine0-fine4 and all"
    )
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-p", "--pretrained_model", help="path of pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject chosen from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    print(args)
    if args.debug:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, current_time)
    else:
        output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    config = Config_MLP()
    # config = Config_Temporal()
    # config = Config_Transformer()
    # config = Config_LSTM()
    save_config(config, output_path)

    dataset = EEGImageNetDataset(args.dataset_dir, args.subject, args.granularity)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = "lambdalabs/sd-image-variations-diffusers"
    feature_extractor = CLIPImageProcessor.from_pretrained(
        diffusion_model, subfolder="feature_extractor", local_files_only=True
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        diffusion_model, subfolder="image_encoder", local_files_only=True
    ).to(device)

    dataset.add_clip_label(
        clip_features_cal(os.path.join(args.dataset_dir, "imageNet_images/"), feature_extractor, image_encoder)
    )
    dataset.use_clip_label = True
    eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
    dataset.add_frequency_feat(de_feat_cal(eeg_data, args.subject, args.granularity))
    train_dataset, test_dataset = get_split_dataset(dataset, config, output_path)
    dataset.use_frequency_feat = True

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    print("----- Creating Model -----", flush=True)

    model = get_eeg_encoder(config)

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
    #     model = nn.DataParallel(model)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.pretrained_model:
        model.load_state_dict(torch.load(os.path.join(args.output_dir, str(args.pretrained_model))))

    print("----- Start training -----", flush=True)

    info_nce_criterion = EEGInfoNCE(temperature=config.temperature, negative_mode="paired")
    mse_criterion = nn.MSELoss()
    mse_ratio = 0.8

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return float(epoch) / float(max(1, config.warmup_epochs))
        return max(config.min_lr / config.lr, 0.98 ** (epoch - config.warmup_epochs))

    scheduler = LambdaLR(optimizer, lr_lambda)

    retrieval = retrievaler(os.path.join(args.dataset_dir, 'imageNet_images'), feature_extractor, image_encoder)

    num_epochs = config.num_epoch
    min_loss = np.finfo(np.float32).max
    max_ndcg = 0
    min_loss_epoch = -1
    min_loss_sim = -1
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            mse_loss = mse_criterion(outputs, labels.squeeze())
            info_nce_loss = info_nce_criterion(outputs, labels.squeeze(), get_negative_samples(labels.squeeze()).to(device))
            optimizer.zero_grad()
            loss = mse_loss * mse_ratio + info_nce_loss * (1-mse_ratio)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
            # torch.cuda.empty_cache()
            # gc.collect()
        train_loss = loss.item()
        scheduler.step()

        model.eval()
        test_loss = 0.0
        cos_sim = 0
        ground_truth_s = 0
        ndcg_s = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_dataloader, desc=f"Testing {epoch + 1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                mse_loss = mse_criterion(outputs, labels.squeeze())
                info_nce_loss = info_nce_criterion(outputs, labels.squeeze(), get_negative_samples(labels.squeeze()).to(device))
                loss = mse_loss * mse_ratio + info_nce_loss * (1-mse_ratio)
                test_loss += loss.item()
                ground_truth, ndcg = retrieval.eval(outputs, labels.squeeze())
                ground_truth_s += ground_truth
                ndcg_s += ndcg
                cos = torch.nn.CosineSimilarity(dim=1)
                sim = cos(outputs, labels.squeeze())
                cos_sim += sim.mean()
                # torch.cuda.empty_cache()
                # gc.collect()

        test_loss /= len(test_dataloader)
        cos_sim /= len(test_dataloader)
        ground_truth_s /= len(test_dataloader)
        ndcg_s /= len(test_dataloader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Cosine Similarity: {cos_sim:.4f}, Ground Truth: {ground_truth_s:.4f}, NDCG: {ndcg_s:.4f}",
            flush=True,
        )
        if test_loss < min_loss:
            min_loss = test_loss
            min_loss_epoch = epoch
            min_loss_sim = cos_sim
            min_loss_ground_truth = ground_truth_s
            min_loss_ndcg = ndcg_s
            torch.save(model.state_dict(), os.path.join(output_path, "eeg_encoder.pth"))
        if ndcg_s > max_ndcg:
            max_loss = test_loss
            max_loss_epoch = epoch
            max_loss_sim = cos_sim
            max_loss_ground_truth = ground_truth_s
            max_ndcg = ndcg_s
    with open(os.path.join(output_path, f"{args.subject}.txt"), "w") as file:
        file.write(f"min loss at epoch {min_loss_epoch}\n")
        file.write(f"Epoch [{min_loss_epoch+1}/{num_epochs}], Test Loss: {min_loss:.4f}, Cosine Similarity: {min_loss_sim:.4f}, Ground Truth: {min_loss_ground_truth:.4f}, NDCG: {min_loss_ndcg:.4f}")
        file.write(f"max ndcg at epoch {max_loss_epoch}\n")
        file.write(f"Epoch [{max_loss_epoch+1}/{num_epochs}], Test Loss: {max_loss:.4f}, Cosine Similarity: {max_loss_sim:.4f}, Ground Truth: {max_loss_ground_truth:.4f}, NDCG: {max_ndcg:.4f}")
