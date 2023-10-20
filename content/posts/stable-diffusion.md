---
title: "Notes on Stable Diffusion"
date: 2023-10-16
author: Nitin
tags: [stable-diffusion]
---

It is not just the LLMs that are all the rage nowadays. Text to image models have also become super powerful and useful. This page attempts to capture some of the concepts involed in the udnerlying method - stable diffusion.


## Overview
Sources:
* https://stable-diffusion-art.com/comfyui/
* https://towardsdatascience.com/the-arrival-of-sdxl-1-0-4e739d5cc6c7

A Stable Diffusion model has three main parts:

MODEL: The noise predictor model in the latent space.
CLIP: The language model preprocesses the positive and the negative prompts.
VAE: The Variational AutoEncoder converts the image between the pixel and the latent spaces.

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



