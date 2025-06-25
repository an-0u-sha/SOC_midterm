# SOC Midterm Report: Image Super-Resolution using SRGAN and ESRGAN

## Overview

This project explores the topic of single image super-resolution (SISR), with a focus on two key models: SRGAN and its improved version, ESRGAN. The goal was to understand how these models work and how they can be implemented and used in practice with real-world datasets and pretrained models.

I worked with both PyTorch and TensorFlow implementations, explored how to use the DIV2K dataset effectively, and learned how to evaluate the output using standard image quality metrics.

## Key Learnings

### 1. SRGAN (Super-Resolution Generative Adversarial Network)

- Introduced the idea of using GANs for generating high-resolution images from low-resolution inputs.
- The generator network learns to upscale images, while the discriminator tries to tell apart real high-res images from the generated ones.
- A key improvement was combining adversarial loss with a content loss based on features extracted from a pretrained VGG network, which helps produce more realistic textures.

### 2. ESRGAN (Enhanced SRGAN)

- Builds on SRGAN by strengthening the generator using Residual-in-Residual Dense Blocks (RRDBs).
- Removes batch normalization layers, which improves both training stability and visual results.
- Introduces a relativistic discriminator that learns to compare the realism of generated and real images, rather than classifying them independently.
- Uses perceptual loss before the activation layers, which helps preserve more fine details and texture.
- Overall, produces sharper and more detailed results compared to SRGAN.

### 3. Working with Pretrained Models (TensorFlow Hub)

- TensorFlow Hub provides easy access to pretrained ESRGAN models, which I used to test super-resolution on sample images.
- Learned the importance of preprocessing (resizing, normalization) and postprocessing (scaling and clipping) steps to make sure the inputs and outputs are correctly handled.
- Using these models is useful for benchmarking and quick prototyping without needing to train from scratch.

### 4. Using the DIV2K Dataset

- DIV2K is a commonly used dataset for image super-resolution, containing paired high-res and low-res images created by downscaling.
- I created a custom PyTorch `Dataset` class to load, crop, and return aligned LR-HR patch pairs.
- This helped me understand how to prepare and manage image data effectively for training super-resolution models.

### 5. Image Quality Evaluation

- Learned how to use tools like the `piq` (Photosynthesis.ImageQuality) library to compute metrics such as PSNR, SSIM, and LPIPS.
- These metrics are essential for evaluating model performance beyond just visual inspection.
- Also experimented with visualizing feature differences and outputs using matplotlib, which helped interpret results more clearly.

## File Descriptions

### `pytorch_esrgan.py`

- Implements a simplified ESRGAN-style generator using PyTorch.
- Includes the RRDB block structure as described in the ESRGAN paper.
- Focuses only on inference using the generator—no training or discriminator is included.
- Helped me understand how the generator is built and how inference works using PyTorch modules.

### `tfhub_esrgan_demo.py`

- Uses a pretrained ESRGAN model from TensorFlow Hub to perform super-resolution on input images.
- Covers image preprocessing, model loading, inference, and displaying output.
- Great for quick experiments and understanding how pretrained models can be applied directly.

### `data_loader_div2k.py`

- Defines a custom PyTorch dataset class to work with the DIV2K dataset.
- Handles reading, aligning, and cropping of LR-HR image pairs.
- Helped me understand how to prepare datasets for training super-resolution models.

## Resources Used

The following resources were instrumental in learning and implementing the concepts in this project:

**GitHub Repositories**
- https://github.com/leverxgroup/esrgan
- https://github.com/lizhuoq/SRGAN

**Tutorials and Documentation**
- TensorFlow Hub ESRGAN Tutorial: https://www.tensorflow.org/hub/tutorials/image_enhancing
- GeeksforGeeks ESRGAN with PyTorch: https://www.geeksforgeeks.org/image-super-resolution-with-esrgan-using-pytorch/

**Papers**
- SRGAN: https://arxiv.org/pdf/1609.04802
- ESRGAN: https://arxiv.org/abs/1809.00219

## Summary

Through this project, I transitioned from a theoretical understanding of super-resolution to applying it in practice. I got hands-on experience with both low-level model building in PyTorch and high-level model usage with TensorFlow Hub. I also learned how to load and prepare real datasets, evaluate image quality using standard metrics, and use academic papers and GitHub implementations to guide my work.
