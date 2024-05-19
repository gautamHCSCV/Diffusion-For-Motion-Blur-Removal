# Motion-Blur-Removal
Use of diffusion model for motion blur removal. Model Architecture </br>
AutoencoderKL (VAE) </br>
UNet2DConditionModel </br>
DDPMScheduler </br>


## AutoencoderKL (VAE)

The AutoencoderKL, a Variational Autoencoder (VAE), plays a crucial role in the encoding and decoding processes of images into and out of a latent space. </br> </br>

Encoding: </br>
Input: A blurry image. </br>
Output: Latent representations (latents). </br> </br>

Decoding: </br>
Input: Latent representations (latents). </br>
Output: Reconstructed sharp image. </br> </br> </br>

## UNet2DConditionModel
The UNet2DConditionModel is a U-Net architecture conditioned on text embeddings. It predicts the noise in the latent space that is added during the diffusion process. It leverages both the input image latents and text embeddings for this task. </br> </br>

Input: </br>
Latents: Encoded blurry image representations. </br>
Timesteps: Indicates the step in the diffusion process. </br>
Encoder Hidden States: Text embeddings generated from the caption "a photo of a sharp image." </br> </br>

Output: </br>
Predicted noise in the latent space. </br> </br> </br>

## DDPMScheduler
The DDPMScheduler manages the diffusion process's schedule, specifying how noise is added and removed over several steps. </br> 

Forward Diffusion Process: Adds noise to images step by step during training. </br>
Reverse Diffusion Process: Removes noise step by step during generation. </br>
In the context of this implementation, the DDPMScheduler determines the timesteps used for training and inference. </br>

## Results
![Screenshot 2024-05-17 092515](https://github.com/gautamHCSCV/Motion-Blur-Removal/assets/65457437/d7298c91-0d87-407e-9ca6-bb27d5e9f3f5)
