import argparse
import os
import json
from tqdm import tqdm
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from PIL import Image
from de_feat_cal import de_feat_cal
from dataset import EEGImageNetDataset
from embedding.utils import load_model, retrievaler


@torch.no_grad()
def reconstruction(
    image_prompt_embed,
    retrieved_images,
    vae,
    unet,
    scheduler,
    feature_extractor,
    image_encoder,
    height,
    width,
    num_inference_steps=50,
    guidance_scale=7.5,
    img2img_strength=0.8,
):
    batch_size = len(image_prompt_embed)
    do_classifier_free_guidance = guidance_scale > 1.0
    device = image_prompt_embed.device

    image_prompt_embed = image_prompt_embed.unsqueeze(1)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(image_prompt_embed)
        image_prompt_embed = torch.cat([negative_prompt_embeds, image_prompt_embed])

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = feature_extractor(
        images=retrieved_images, return_tensors="pt", size=(512, 512), crop_size=(512, 512)
    ).pixel_values.to(device)
    with torch.no_grad():
        latents = vae.encode(latents).latent_dist.sample() * vae.config.scaling_factor
    init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]
    latent_timestep = timesteps[:1].repeat(batch_size)
    noise = torch.randn([1, 4, 64, 64], device=device)
    latents = scheduler.add_noise(latents, noise, torch.LongTensor(latent_timestep.cpu()))

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=image_prompt_embed).sample
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    image = ((image / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", required=True, help="directory name of EEG-ImageNet dataset path")
    parser.add_argument(
        "-g", "--granularity", required=True, help="granularity chosen from coarse, fine0-fine4 and all"
    )
    parser.add_argument("-m", "--model", required=True, help="model")
    parser.add_argument("-b", "--batch_size", default=40, type=int, help="batch size")
    parser.add_argument("-p", "--pretrained_model", help="path of pretrained model")
    parser.add_argument("-s", "--subject", default=0, type=int, help="subject chosen from 0 to 15")
    parser.add_argument("-o", "--output_dir", required=True, help="directory to save results")
    args = parser.parse_args()
    print(args)

    dataset = EEGImageNetDataset(args.dataset_dir, args.subject, args.granularity)
    # extract frequency domain features
    eeg_data = np.stack([i[0].numpy() for i in dataset], axis=0)
    de_feat = de_feat_cal(eeg_data, args.subject, args.granularity)
    dataset.add_frequency_feat(de_feat)
    dataset.use_frequency_feat = True
    dataset.use_image_label = True
    test_index = np.array([i for i in range(len(dataset)) if i % 50 > 29])
    test_subset = Subset(dataset, test_index)
    dataloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    diffusion_model = "lambdalabs/sd-image-variations-diffusers"
    feature_extractor = CLIPImageProcessor.from_pretrained(
        diffusion_model, subfolder="feature_extractor", local_files_only=True
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        diffusion_model, subfolder="image_encoder", local_files_only=True
    ).to(device)
    image_retrieval = retrievaler("../data/imageNet_images_all/", feature_extractor, image_encoder)

    vae = AutoencoderKL.from_pretrained(diffusion_model, subfolder="vae", local_files_only=True).to(device)
    unet = UNet2DConditionModel.from_pretrained(diffusion_model, subfolder="unet", local_files_only=True).to(device)
    scheduler = PNDMScheduler.from_pretrained(diffusion_model, subfolder="scheduler", local_files_only=True)
    eeg_encoder = load_model(args.pretrained_model).to(device)
    eeg_encoder.eval()
    
    retrieval_list = []
    with torch.no_grad():
        for index, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device=device)
            image_prompt_embed = eeg_encoder(inputs)
            retrieved_images, D, I = image_retrieval.retrieval(image_prompt_embed, k=10)
            retrieved_images = [row[0] for row in retrieved_images]
            print(f"{index + 1} / {len(dataloader)}, retrieved: {[image_retrieval.image_files[row[0]] for row in I]}")
            retrieval_list.extend([image_retrieval.image_files[row[0]] for row in I])
            images_gen = reconstruction(
                image_prompt_embed.to(device),
                retrieved_images,
                vae,
                unet,
                scheduler,
                feature_extractor,
                image_encoder,
                512,
                512,
                20,
                7.5,
                0.9,
            )
            for i, image in enumerate(images_gen):
                image_pil = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
                image_pil.save(
                    os.path.join(
                        args.output_dir, f"generated_s{args.subject}_big", f"{i + 1 + index * args.batch_size}.png"
                    )
                )
        with open(os.path.join(args.output_dir, f"generated_s{args.subject}_big", "retrieval_list.json"), "w") as f:
            json.dump(retrieval_list, f)
