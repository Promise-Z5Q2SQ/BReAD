# 🍞 BReAD-anonymous 

This is the official repository for the paper "**Brain Image Reconstruction with Retrieval-Augmented Diffusion**".

We propose 🍞 **BReAD** (**B**rain Image Reconstruction with **Re**trieval-**A**ugmented **D**iffusion), 
a framework for visual reconstruction from EEG/MEG signals with a specially designed retrieval-augmented diffusion model. 
Figure 1 shows how our method differs from previous approaches by introducing a retrieval-augmented diffusion process.

<img width="588" alt="intro" src="https://github.com/user-attachments/assets/9bf91098-7b22-4772-b652-09190e620f02" />

**Figure 1**: Illustration of the proposed method compared to the previous method. While previous methods directly map brain signals to the latent space and decode the result, our framework retrieves semantically relevant priors and refines them through a conditioned diffusion process guided by brain embeddings before decoding into final reconstruction.

<img width="1011" alt="procedure" src="https://github.com/user-attachments/assets/41d399fe-64ed-4311-90a2-d7d6268a135a" />

**Figure 2**: The main procedure of the proposed framework BreAD. 
  (A) Brain Encoder transforms brain signals into embeddings aligned with a shared representation space using contrastive learning. 
  (B) Brain embeddings are used in the Brain Image Retrieval module to retrieve semantically relevant images from a large-scale image dataset as priors for image reconstruction. 
  (C) The retrieved priors are inputted into a Diffusion Pipeline, where forward diffusion introduces noise to balance prior information with flexibility, and reverse diffusion iteratively generates high-quality reconstructed images conditioned on the brain embeddings and retrieved priors.

<img width="907" alt="reconstruction" src="https://github.com/user-attachments/assets/8077eccc-9bad-4df7-b5f7-b589bb769c6c" />

**Figure 3**: Reconstruction results sampled from subject S8 of the EEG-ImageNet dataset, including ground-truth images, retrieved priors, reconstructed images of BReAD, and reconstructed images of baseline~\cite{high_takagi}. 
  (a) presents good cases where the reconstruction effectively captures the semantic content of the objects. 
  (b) presents bad cases where low-level details of the objects, such as orientation, quantity, and color, still exhibit some flaws.

<img width="801" alt="image" src="https://github.com/user-attachments/assets/28c1f487-77bb-4042-9fa0-0f4b40cba64e" />

**Figure 4**: Reconstruction results for fine-grained categories of "dog" in the EEG-ImageNet dataset, showing ground-truth images, retrieved priors, and reconstructed images.
