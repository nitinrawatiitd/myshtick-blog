---
title: "Notes on Stable Diffusion"
date: 2023-10-16
author: Nitin
tags: [stable-diffusion][vision]
---

It is not just the LLMs that are all the rage nowadays. Text to image models have also become super powerful and useful. This page attempts to capture some of the concepts involed in the udnerlying method - stable diffusion.


## Overview
Sources:
* https://stable-diffusion-art.com/comfyui/
* https://towardsdatascience.com/the-arrival-of-sdxl-1-0-4e739d5cc6c7
* Deep learning foundations part 2 from Jeremy Howard


## Concepts
A Stable Diffusion model has three main parts:

MODEL: The noise predictor model in the latent space, usually a Unet
CLIP: The language model that interprets the input text and guide image generation
VAE: The Variational AutoEncoder converts the image between the pixel and the compressed latent space.

We'll discuss about the intuition behind each part and what role it plays in the diffusion model

### Intuition behind stable diffusion
This is a great blog from lesson 9 of Jeremy Howard's deep learning foundation course, that explains the intuition - https://rekil156.github.io/rekilblog/posts/lesson9_stableDissufion/Lesson9.html. Most notes are derived from there.

Recap:
* Given that there is a magical function/API that can tell us the probability of an input image being a handwritten digit P(X), we can learn to change the pixels of any input image so that is closer to a handwritten image. So we are not modifying the weights of a model. We are modifying the pixels.
* To find out which pixels to change and by how much, we can calculate the gradients with respect to each pixel i in the image $\frac{\partial P(X_i)}{\partial X_i}$. For all the pixels that make up X (vector of pixels), the partial derivative wrt X can be collectively represented by $\nabla_x$ - called del or nabla. In our case since we are measuring change in Probability wrt X, we can also write it as $\nabla_x P$. These gradients are also populary called the **score functions**.
* To built this function/API we need to train a model that can identify how close an image is to a handwritten digit. We create the training data by using real handwritten digits and then just chucking random noise on top of it. **It’s a little bit awkward for us to come up with an exact score which can tell us how much these noisy images are like a handwritten digit so instead let’s predict how much noise was added**. So we create a neural net for which we need:
    * Inputs - noisy digits
    * Outputs - noise
    * loss function - MSE, between the predicted output(noise) and the actual noise
* Once we train the neural net we can pass it an image (random noise) and it’s going to spit out information saying which part of that image it think’s is noise. We can iteratively remove the noise and reach to an image that looks like a handwritten digit. [Why do we remove noise iteratively?](#why-do-we-remove-noise-iteratively) - we'll touch on that later.

### UNET, VAE and CLIP - building blocks of stable diffusion

There are 3 compenents to most stable diffusion models:
* Unet - This is the deep learning model that learns to identify noise in an image.
    * Input - somewhat noisy image, it could be no noisy at all or it could be all noise
    * Output - noise
    But images are quite big in size and can take a lot if time to train. That's where VAE's come in that can compress the image into latent dimensions that are more manageable. Stable diffusion in latent space is also called latent diffusion.
* VAE - This is made of an encoder that converts the input image to a latent image (fewer dimensions) and a decoder that converts a latent representation back to an image. Note: A latest model, Matryoshka Diffusion Models (MDM), skips VAE and does modeling directly in the pixel space, albeit more effeciently than previous approaches.
    * Encoder
        * Input - pixels
        * Output - latent representation
    * Decoder
        * Input - latent representation
        * Output - pixels

* CLIP - This is to model how text can be used for guiding image generation. There are pair of models, one model which is a text encoder and one model which is an image encoder, that are trained together using what is called Contrastive Language-Image Pre-training (CLIP). So we can pass the image to the image encoder and text to the text encoder and they will each give us two embeddings (same dimensions). The ideas is to train a model that'll make embeddings for text and image that describe the same things to be more similar and that don't to be disimilar.
    * Input - text
    * Output - embedding

![](/img/CLIP.png "CLIP")


## Why do we remove noise iteratively?

Visual way to understand the intuition behind why the denoising steps need to happen gradually and not in one go.

![](/img/why_incremental_denoising.png "Source: Lesson 9A 2022 - Stable Diffusion deep dive")

There are parallelss between the denoising in latent diffusion models and the gradient descent we have normally come to understand in the neural network models. Most of the tricks and methods used there can be used in stable diffusion as well.

## Classifier free guidance

You'll often encounter the term classifier free guidance. What does it exactly mean?

Source: https://sander.ai/2022/05/26/guidance.html

## Use of blank text in diffusion models


## Popular models

We'll focus on models from stability.ai. There are 2:

**Stable diffusion 1.5 (and higher versions)**
![](/img/SD1_5.webp "Usual SD 1.5 generation pipeline")

**SDXL**
![](/img/SDXL.webp "SDXL generation pipeline")

## History

## Popular Web UIs
* ComfyUI
* Kohya
* Automatic1111



