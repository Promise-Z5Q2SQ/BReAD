# 🍞 BReAD-anonymous 

This is the official repository for the paper "**Brain Image Reconstruction with Retrieval-Augmented Diffusion**".

We propose 🍞 **BReAD** (**B**rain Image Reconstruction with **Re**trieval-**A**ugmented **D**iffusion), 
a framework for visual reconstruction from EEG/MEG signals with a specially designed retrieval-augmented diffusion model. 
Figure 1 shows how our method differs from previous approaches by introducing a retrieval-augmented diffusion process.

{todo}

**Figure 1**: Illustration of the proposed method compared to the previous method. While previous methods directly map brain signals to the latent space and decode the result, our framework retrieves semantically relevant priors and refines them through a conditioned diffusion process guided by brain embeddings before decoding into final reconstruction.

{todo}

**Figure 2**: The main procedure of the proposed framework BreAD. 
  (A) Brain Encoder transforms brain signals into embeddings aligned with a shared representation space using contrastive learning. 
  (B) Brain embeddings are used in the Brain Image Retrieval module to retrieve semantically relevant images from a large-scale image dataset as priors for image reconstruction. 
  (C) The retrieved priors are inputted into a Diffusion Pipeline, where forward diffusion introduces noise to balance prior information with flexibility, and reverse diffusion iteratively generates high-quality reconstructed images conditioned on the brain embeddings and retrieved priors.

{todo}

**Figure 3**: Reconstruction results sampled from subject S8 of the EEG-ImageNet dataset, including ground-truth images, retrieved priors, reconstructed images of BReAD, and reconstructed images of baseline~\cite{high_takagi}. 
  (a) presents good cases where the reconstruction effectively captures the semantic content of the objects. 
  (b) presents bad cases where low-level details of the objects, such as orientation, quantity, and color, still exhibit some flaws.


{todo}

**Figure 4**: Reconstruction results for fine-grained categories of "dog" in the EEG-ImageNet dataset, showing ground-truth images, retrieved priors, and reconstructed images.