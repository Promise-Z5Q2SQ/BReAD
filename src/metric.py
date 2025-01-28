import numpy as np
import os
from PIL import Image
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import random
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


# 计算PixCorr（皮尔逊相关系数）
def pixcorr(image1, image2):
    # 扁平化图像
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    # 计算皮尔逊相关系数
    corr, _ = pearsonr(image1_flat, image2_flat)
    return corr


def calculate_ssim(image1, image2):
    return ssim(image1, image2, channel_axis=0, data_range=1.0)


# 读取图像并预处理
def load_and_preprocess_image(image_path, mean=None, std=None, size=512, crop_size=(512, 512)):
    image = Image.open(image_path).convert("RGB")  # 确保是RGB图像
    width, height = image.size
    if width < height:
        # 宽度是短边，调整宽度为 512
        new_width = size
        new_height = int((new_width / width) * height)  # 按比例调整高度
    else:
        # 高度是短边，调整高度为 512
        new_height = size
        new_width = int((new_height / height) * width)  # 按比例调整宽度

    if mean and std:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((new_width, new_height)),
                transforms.CenterCrop(crop_size),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((new_width, new_height)),
                transforms.CenterCrop(crop_size),
            ]
        )
    pixel_values = preprocess(image)

    return pixel_values


def get_clip_embedding(batch_paths, *args, **kwargs):
    feature_extractor = kwargs["feature_extractor"]
    image_encoder = kwargs["image_encoder"]
    device = image_encoder.device
    images = [Image.open(img_path).convert("RGB") for img_path in batch_paths]
    with torch.no_grad():
        processed = feature_extractor(images=images, return_tensors="pt").pixel_values.to(device)
        features = image_encoder(processed).image_embeds
    return features.cpu().numpy()  # (768)


def get_alexnet_embedding(batch_paths, layer_num, *args, **kwargs):
    device = kwargs["device"]
    # 加载并预处理图像
    image_list = []
    for image_path in batch_paths:
        image = load_and_preprocess_image(
            image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=256, crop_size=(224, 224)
        ).to(device)
        image_list.append(image)
    # 加载预训练的 AlexNet 模型
    alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
    alexnet.eval()  # 设置为评估模式
    feat_extractor = create_feature_extractor(alexnet, return_nodes=[f"features.{layer_num}"])
    with torch.no_grad():
        embedding = feat_extractor(torch.stack(image_list))[f"features.{layer_num}"].cpu().numpy()
        embedding = embedding.reshape(embedding.shape[0], -1)
    return embedding  # alex2:(32448), alex5:(9126)


def get_inception_embedding(batch_paths, *args, **kwargs):
    device = kwargs["device"]
    # 加载并预处理图像
    image_list = []
    for image_path in batch_paths:
        image = load_and_preprocess_image(
            image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=299, crop_size=(299, 299)
        ).to(device)
        image_list.append(image)
    # 加载 InceptionV3 模型
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
    model.eval()
    feat_extractor = create_feature_extractor(model, return_nodes=["avgpool"])
    with torch.no_grad():
        embedding = feat_extractor(torch.stack(image_list))["avgpool"].cpu().numpy()
        embedding = embedding.reshape(embedding.shape[0], -1)
    return embedding  # (2048)


# 计算图片的 EfficientNet 表示
def get_efficientnet_embedding(batch_paths, *args, **kwargs):
    device = kwargs["device"]
    # 加载并预处理图像
    image_list = []
    for image_path in batch_paths:
        image = load_and_preprocess_image(
            image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=240, crop_size=(240, 240)
        ).to(device)
        image_list.append(image)
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT).to(device)
    model.eval()
    feat_extractor = create_feature_extractor(model, return_nodes=["avgpool"])
    with torch.no_grad():
        embedding = feat_extractor(torch.stack(image_list))["avgpool"].cpu().numpy()
        embedding = embedding.reshape(embedding.shape[0], -1)
    return embedding  # (1280)


def get_swav_embedding(batch_paths, *args, **kwargs):
    device = kwargs["device"]
    image_list = []
    for image_path in batch_paths:
        image = load_and_preprocess_image(
            image_path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], size=240, crop_size=(240, 240)
        ).to(device)
        image_list.append(image)
    model = torch.hub.load("facebookresearch/swav:main", "resnet50").to(device)
    model.eval()
    feat_extractor = create_feature_extractor(model, return_nodes=["avgpool"])
    with torch.no_grad():
        embedding = feat_extractor(torch.stack(image_list))["avgpool"].cpu().numpy()
        embedding = embedding.reshape(embedding.shape[0], -1)
    return embedding  # (2048)


# 主函数，根据输入参数返回对应的 embedding
def get_image_embeddings(image_paths, model_name, batch_size=200, *args, **kwargs):
    print(model_name, batch_size)
    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i : i + batch_size]
        if model_name == "clip":
            batch_embeddings = get_clip_embedding(batch_paths, *args, **kwargs)
        elif model_name == "alex2":
            batch_embeddings = get_alexnet_embedding(batch_paths, layer_num=5, *args, **kwargs)
        elif model_name == "alex5":
            batch_embeddings = get_alexnet_embedding(batch_paths, layer_num=12, *args, **kwargs)
        elif model_name == "inception":
            batch_embeddings = get_inception_embedding(batch_paths, *args, **kwargs)
        elif model_name == "eff":
            batch_embeddings = get_efficientnet_embedding(batch_paths, *args, **kwargs)
        elif model_name == "swav":
            batch_embeddings = get_swav_embedding(batch_paths, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings, axis=0)


# 计算 Two-way identification metric
def two_way_identification_metric(
    generated_images_dir,
    real_images_dir,
    correspondences_file,
    indices,
    model_name,
    num_comparisons=500,
    *args,
    **kwargs,
):
    # 读取图像文件对应关系
    with open(correspondences_file, "r") as f:
        correspondences = [line.strip() for line in f.readlines()]

    correct_count = 0
    total_count = 0
    generated_image_paths = [os.path.join(generated_images_dir, f"{index+1}.png") for index, i in enumerate(indices)]
    real_image_paths = [
        os.path.join(real_images_dir, correspondences[i].split("_")[0], correspondences[i]) for i in indices
    ]
    # 批量获取生成图像和真实图像的 embedding
    generated_embeddings = get_image_embeddings(generated_image_paths, model_name, *args, **kwargs)
    real_embeddings = get_image_embeddings(real_image_paths, model_name, *args, **kwargs)

    embeddings = []
    for i in range(len(indices)):
        embeddings.append((generated_embeddings[i], real_embeddings[i]))

    # 对每个图像进行 Two-way Identification
    for i in tqdm(range(len(embeddings))):
        # 当前真实图像和生成图像的 embedding
        generated_image_embedding, real_image_embedding = embeddings[i]
        correct = 0
        # 从其他所有图像中随机选择 num_comparisons 张进行比较
        other_indices = [j for j in range(len(embeddings)) if j != i]
        random_other_indices = random.sample(other_indices, num_comparisons)

        for j in random_other_indices:
            # 获取另外一张图像的生成图像的 embedding
            other_generated_image_embedding, other_real_image_embedding = embeddings[j]
            # 计算当前真实图像与生成图像之间的皮尔逊相关系数
            corr_with_generated = pearsonr(real_image_embedding, generated_image_embedding)[0]
            # 计算当前真实图像与另一生成图像之间的皮尔逊相关系数
            corr_with_other_generated = pearsonr(real_image_embedding, other_generated_image_embedding)[0]
            # 进行比较
            if corr_with_generated > corr_with_other_generated:
                correct += 1

        # 计算正确率
        correct_count += correct
        total_count += num_comparisons  # 每次与 num_comparisons 张图像进行比较

    # 计算最终 Two-way identification 的平均正确率
    accuracy = correct_count / total_count
    return accuracy


# 计算 Average Correlation Distance
def calculate_acd_metric(
    generated_images_dir, real_images_dir, correspondences_file, indices, model_name, *args, **kwargs
):
    # 读取图像文件对应关系
    with open(correspondences_file, "r") as f:
        correspondences = [line.strip() for line in f.readlines()]

    generated_image_paths = [os.path.join(generated_images_dir, f"{index+1}.png") for index, i in enumerate(indices)]
    real_image_paths = [
        os.path.join(real_images_dir, correspondences[i].split("_")[0], correspondences[i]) for i in indices
    ]

    # 批量获取生成图像和真实图像的 embedding
    generated_embeddings = get_image_embeddings(generated_image_paths, model_name, *args, **kwargs)
    real_embeddings = get_image_embeddings(real_image_paths, model_name, *args, **kwargs)

    # 计算相关性距离
    correlation_distances = []
    for i in tqdm(range(len(indices))):
        corr, _ = pearsonr(generated_embeddings[i], real_embeddings[i])
        correlation_distances.append(1 - corr)  # Pearson correlation distance: 1 - correlation

    return np.mean(correlation_distances)


# 计算一组图像的重建度量
def calculate_metrics(
    generated_images_dir, real_images_dir, correspondences_file, indices, feature_extractor, image_encoder
):
    # 读取图像文件对应关系
    with open(correspondences_file, "r") as f:
        correspondences = [line.strip() for line in f.readlines()]

    # 计算 PixCorr 和 SSIM
    pixcorr_values = []
    ssim_values = []
    # 遍历指定的生成图像和真实图像
    for index, i in enumerate(tqdm(indices)):
        # 加载生成图像
        generated_image_path = os.path.join(generated_images_dir, f"{index+1}.png")
        generated_image_pixel_values = load_and_preprocess_image(generated_image_path).numpy()
        # 预处理真实图像
        real_image_path = os.path.join(real_images_dir, correspondences[i].split("_")[0], correspondences[i])
        real_image_pixel_values = load_and_preprocess_image(real_image_path).numpy()

        pixcorr_value = pixcorr(generated_image_pixel_values, real_image_pixel_values)
        pixcorr_values.append(pixcorr_value)
        ssim_value = calculate_ssim(generated_image_pixel_values, real_image_pixel_values)
        ssim_values.append(ssim_value)
    avg_pixcorr = np.mean(pixcorr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Average PixCorr: {avg_pixcorr}")
    print(f"Average SSIM: {avg_ssim}")

    # 计算 Two-way Identification Metric
    for model_name in ["alex2", "alex5", "inception", "clip"]:
        two_way_identification_accuracy = two_way_identification_metric(
            generated_images_dir,
            real_images_dir,
            correspondences_file,
            indices,
            model_name,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            device=image_encoder.device,
        )
        print(f"Two-way Identification Accuracy of {model_name}: {two_way_identification_accuracy}")

    # 计算 Average Correlation Distance
    for model_name in ["eff", "swav"]:
        acd_metric = calculate_acd_metric(
            generated_images_dir,
            real_images_dir,
            correspondences_file,
            indices,
            model_name,
            device=image_encoder.device,
        )
        print(f"Average Correlation Distance of {model_name}: {acd_metric}")


if __name__ == "__main__":
    generated_images_dir = "../output/generated_s5_big/"
    real_images_dir = "../data/imageNet_images/"
    correspondences_file = "../output/s5_path.txt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", subfolder="feature_extractor", local_files_only=True
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers", subfolder="image_encoder", local_files_only=True
    ).to(device)

    # indices = np.arange(0, 4000)
    indices = np.array([i for i in range(4000) if i % 50 > 29])
    # indices = np.array([i for i in range(4000) if i % 50 < 30])

    print(f"Calculating metrics of {generated_images_dir} and {correspondences_file}")
    calculate_metrics(
        generated_images_dir, real_images_dir, correspondences_file, indices, feature_extractor, image_encoder
    )
