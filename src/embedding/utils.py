import torch
from .model import get_eeg_encoder
from .config import *
import json
from tqdm import tqdm
import os
import numpy as np
import faiss
from transformers import AutoModel, AutoProcessor
from PIL import Image
from typing import List, Tuple
import random


def load_model(model_path="src/embedding/local_model"):
    config_path = os.path.join(model_path, "config.json")
    model_path = os.path.join(model_path, "eeg_encoder.pth")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = Config()
    config.load_config(config_dict)

    model_checkpoint = torch.load(model_path)
    if "module." in list(model_checkpoint.keys())[0]:
        model_checkpoint = {k.replace("module.", ""): v for k, v in model_checkpoint.items()}
    model = get_eeg_encoder(config)
    model.load_state_dict(model_checkpoint)

    return model


class retrievaler:
    def __init__(self, image_path, feature_extractor, image_encoder, max_retrieval_depth=50, size="all"):
        self.image_path = image_path
        self.max_retrieval_depth = max_retrieval_depth
        self.image_files = []
        self.document_list = []
        device = image_encoder.device

        print("------ Loading image files ------")
        if os.path.exists(os.path.join(image_path, "image_files.json")):
            with open(os.path.join(image_path, "image_files.json"), "r") as f:
                self.image_files = json.load(f)
        else:
            for root, dirs, files in os.walk(self.image_path):
                for file in files:
                    if file.lower().endswith((".jpeg")):
                        self.image_files.append(file.split(".")[0])
            with open(os.path.join(image_path, "image_files.json"), "w") as f:
                json.dump(self.image_files, f)
        n_images = len(self.image_files)
        print(f"total image files: {n_images}")
        if size == "all":
            selected_indices = list(range(len(self.image_files)))
        else:
            selected_indices = random.sample(range(len(self.image_files)), size)
            self.image_files = [self.image_files[i] for i in selected_indices]
        print(f"selected image files: {len(self.image_files)}")

        batch_size = 200
        for i in tqdm(range((0) * batch_size, len(self.image_files), batch_size)):
            image_batch = self.image_files[i : i + batch_size]
            # 加载并预处理图片
            for image_file in image_batch:
                image_path = os.path.join(self.image_path, image_file.split("_")[0], image_file + ".JPEG")
                image = Image.open(image_path)
        print("Test")

        print("------ Setting up document list ------")
        clip_path = os.path.join(image_path, "clip_features.dat")
        if not os.path.exists(clip_path):
            clip_tmp_path = os.path.join(image_path, "clip_features.txt")
            batch_size = 100  # 文件太大，分批处理
            with torch.no_grad():
                for i in tqdm(range((0) * batch_size, len(self.image_files), batch_size)):
                    clip_dict = {}
                    images = []
                    image_file_names = []
                    image_batch = self.image_files[i : i + batch_size]
                    # 加载并预处理图片
                    for image_file in image_batch:
                        image_path = os.path.join(self.image_path, image_file.split("_")[0], image_file + ".JPEG")
                        image = Image.open(image_path)
                        images.append(image)
                        image_file_names.append(image_file + ".JPEG")
                    # 批量预处理
                    processed_batch = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
                    # 批量编码得到 embeddings
                    features_batch = image_encoder(processed_batch).image_embeds.cpu().squeeze().numpy().tolist()
                    # 将结果保存到 self.document_list 和 clip_dict
                    self.document_list.extend(features_batch)
                    for idx, image_file_name in enumerate(image_file_names):
                        clip_dict[image_file_name] = features_batch[idx]
                    # 保存到文件
                    with open(clip_tmp_path, "a") as f:
                        json.dump(clip_dict, f)
                        f.write("\n")  # 在每个对象后添加换行符

            with open(clip_tmp_path, "r") as file:
                # total_lines = sum(1 for _ in file)
                total_lines = 84449  # tmp
                file.seek(0)  # 重置文件指针
                tmp_list = []
                self.document_list = np.memmap(
                    clip_path, dtype=np.float32, mode="w+", shape=(len(self.image_files), 768)
                )
                for line_number, line in enumerate(tqdm(file, total=total_lines, desc="each line"), start=1):
                    data = json.loads(line.strip())
                    tmp_list.extend(list(data.values()))
                    if line_number % 10000 == 0:
                        values = np.empty((batch_size * 10000, 768), dtype=np.float32)
                        for i, row in enumerate(tqdm(tmp_list, desc=f"{line_number // 10000}th batch")):
                            values[i] = row
                        self.document_list[(line_number - 10000) * batch_size : line_number * batch_size] = values
                        self.document_list.flush()  # 刷新到磁盘
                        tmp_list = []
                    if line_number == total_lines:
                        values = np.empty((444819, 768), dtype=np.float32)
                        for i, row in enumerate(tqdm(tmp_list, desc=f"last batch")):
                            values[i] = row
                        self.document_list[(line_number - 4449) * batch_size : len(self.image_files)] = values
                        self.document_list.flush()  # 刷新到磁盘
                        tmp_list = []
                print(f"Shape of clip_array: {self.document_list.shape}")
        else:
            if size == "all":
                self.document_list = np.memmap(clip_path, dtype=np.float32, mode="r", shape=(n_images, 768))
            else:
                self.all_clip_array = np.memmap(clip_path, dtype=np.float32, mode="r", shape=(n_images, 768))
                print(f"Shape of total clip_array: {self.all_clip_array.shape}")
                self.document_list = self.all_clip_array[selected_indices]
            print(f"Shape of selected clip_array: {self.document_list.shape}")

        print("------ Setting up faiss index ------")
        self.index = faiss.IndexFlatL2(self.document_list.shape[1])
        print(f"index trained: {self.index.is_trained}")
        self.index.add(self.document_list)
        print(f"total document: {self.index.ntotal}")

    def __del__(self):
        del self.all_clip_array

    def retrieval(self, query, k=10) -> List[Image.Image]:
        query = query.cpu().numpy()
        D, I = self.index.search(query, k)
        images = [[self[x] for x in row] for row in I]
        return images, D, I

    def eval(self, outputs, labels) -> Tuple[float, float]:
        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        ground_truth = []
        ndcg_s = []
        # outputs.shape = [batch_size, 768]
        batch_size = outputs.shape[0]

        # 计算labels在document_list中的index
        labels_numpy = labels.cpu().numpy()
        D, I = self.index.search(labels_numpy, 1)
        labels_index = I.squeeze()

        D, I = self.index.search(outputs.cpu().numpy(), int(self.index.ntotal / 2))

        for i in range(batch_size):
            # 查询labels_index[i]在I[i]中的位置，若没有则记为int(self.index.ntotal / 2)
            image_index = labels_index[i]
            image_index_in_I = list(I[i]).index(image_index) if image_index in I[i] else int(self.index.ntotal / 2)
            ground_truth.append(image_index_in_I)

            # 计算NDCG，用cosine similarity当作relevance
            DCG = 0
            cos = torch.nn.CosineSimilarity(dim=0)
            for j in range(self.max_retrieval_depth):
                DCG += cos(labels[i], torch.tensor(self.document_list[I[i][j]]).to(device)).item() / np.log2(j + 2)
            IDCG = 0
            for j in range(self.max_retrieval_depth):
                IDCG += 1 / np.log2(j + 2)
            NDCG = DCG / IDCG
            ndcg_s.append(NDCG)

        return np.mean(ground_truth), np.mean(ndcg_s)

    def __getitem__(self, idx) -> Image:
        image = Image.open(
            os.path.join(self.image_path, self.image_files[idx].split("_")[0], self.image_files[idx] + ".JPEG")
        )
        return image


if __name__ == "__main__":
    model = load_model()
    print(model.forward(torch.rand(1, 62, 400)))
    image_path = "data/imageNet_images"
    doc_list = retrievaler(image_path)
    query = torch.rand(768)
    images = doc_list.retrieval(query)
    print(images)
