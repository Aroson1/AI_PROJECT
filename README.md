# Fast Neural Style Transfer - CSE 311 AI Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aroson1/AI_PROJECT/blob/master/notebooks/Colab_Train_And_Run.ipynb)

**Project Title:** Real-Time Artistic Image Stylization Using Deep Neural Networks

A lightweight implementation of Fast Neural Style Transfer optimized for Google Colab Free Tier. This project enables training and applying artistic styles to images in real-time using deep convolutional neural networks.

<p align="center">
  <img src="assets/animate.gif" width="650" alt="Style Transfer Animation">
</p>

## 📋 Project Overview

Neural Style Transfer (NST) is a deep learning technique that applies the artistic style of one image to the content of another. This implementation uses a feed-forward convolutional neural network with residual blocks for real-time stylization, trained with perceptual losses computed using a pre-trained VGG19 network.

## ✨ Key Features

- **Fast Training**: Optimized for Colab Free Tier (2-3 epochs, ~2000 images)
- **Lightweight Architecture**: Feed-forward CNN with residual blocks
- **Efficient Loss Computation**: VGG19-based perceptual losses (content + style + TV)
- **Real-time Inference**: Stylize images in seconds
- **Easy to Use**: Single Colab notebook for complete workflow
- **Educational**: Well-documented code perfect for learning

## ## 🏗️ Architecture

### Generator Network (TransformerNet)
- **Encoder**: 3 convolutional layers (downsampling)
- **Transformer**: 5 residual blocks  
- **Decoder**: 3 convolutional layers (upsampling)
- Total parameters: ~1.6M

### Loss Function (VGG19-based)
- **Content Loss**: MSE on relu3_3 features (weight: 1.0)
- **Style Loss**: Gram matrix MSE on relu1_2, relu2_2, relu3_3, relu4_3 (weight: 10.0)
- **Total Variation Loss**: Spatial smoothness (weight: 1e-6)

## 📦 Repository Structure

```
AI_Project/
├── src/                          # Core source code
│   ├── transformer.py            # Generator network
│   ├── vgg_loss.py              # VGG19 loss functions
│   ├── datasets.py              # Dataset loader
│   ├── train.py                 # Training script
│   └── inference.py             # Inference script
├── notebooks/
│   └── Colab_Train_And_Run.ipynb  # Main notebook
├── models/checkpoints/           # Saved models (git ignored)
├── requirements-min.txt          # Minimal dependencies
└── README.md                     # This file
```

## 🚀 Quick Start - Google Colab

**Easiest Method** (Recommended):

1. Click the **"Open in Colab"** badge at the top
2. Run all cells in the notebook
3. The notebook automatically:
   - Installs dependencies
   - Downloads COCO dataset subset
   - Trains the model (2-3 epochs, ~15-20 minutes)
   - Plots loss curves
   - Generates stylized images

## 💻 Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements-min.txt

# Download COCO dataset (validation set)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

## 🎨 Usage

### Training

```bash
python src/train.py \
  --dataset-path val2017 \
  --style-image picasso_selfportrait.jpg \
  --epochs 3 \
  --batch-size 4 \
  --image-size 256 \
  --subset-size 2000
```

**Training Parameters:**
- `--epochs`: Number of epochs (default: 3)
- `--batch-size`: Batch size (default: 4, adjust for your GPU)
- `--image-size`: Image resolution (default: 256)
- `--subset-size`: Number of training images (default: 2000)
- `--lr`: Learning rate (default: 1e-3)
- `--style-weight`: Style loss weight (default: 10.0)
- `--content-weight`: Content loss weight (default: 1.0)

### Inference

**Single Image:**
```bash
python src/inference.py \
  --checkpoint models/checkpoints/final_model.pth \
  --input path/to/content/image.jpg \
  --output stylized_output.jpg
```

**Batch Processing:**
```bash
python src/inference.py \
  --checkpoint models/checkpoints/final_model.pth \
  --input input_directory/ \
  --output output_directory/ \
  --batch
```

## 📊 Training Results

Typical results on Colab Free Tier (T4 GPU):
- **Training time**: ~5-6 minutes per epoch
- **Total training time**: ~15-20 minutes (3 epochs)
- **Inference time**: ~0.5 seconds per image
- **Model size**: ~6.5 MB

## 📚 Requirements

Minimal dependencies (see `requirements-min.txt`):
- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Pillow >= 8.0.0
- matplotlib >= 3.3.0
- tqdm >= 4.60.0
- numpy >= 1.19.0
- opencv-python >= 4.5.0

## 🎯 Project Details - CSE 311

**Course**: CSE 311 - Artificial Intelligence  
**Project Title**: Real-Time Artistic Image Stylization Using Deep Neural Networks

**Objectives:**
1. Implement Fast Neural Style Transfer for real-time image stylization
2. Optimize training for limited computational resources (Colab Free Tier)
3. Achieve artistic style transfer while preserving content structure
4. Demonstrate practical deep learning application in computer vision

**Dataset**: COCO 2017 (subset of 2,000 images at 256×256 resolution)

## 🔗 Important Links

- **COCO Dataset**: http://images.cocodataset.org/zips/val2017.zip
- **Original Paper**: [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155)
- **VGG19 Architecture**: [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)

## 📖 References

1. Johnson et al. - [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
2. Gatys et al. - [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
3. [PyTorch Style Transfer Tutorial](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
4. [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)

## 📄 License

This project is for educational purposes as part of CSE 311 coursework.

## 🙏 Acknowledgments

- Original implementation inspired by various Fast Neural Style Transfer implementations
- COCO dataset from Microsoft COCO: Common Objects in Context
- Pre-trained VGG19 from PyTorch Model Zoo

---

**Note**: This is a simplified, educational implementation optimized for Google Colab Free Tier. For production use, consider training with more epochs, higher resolution images, and additional optimizations.

## The Problem: 
Each new content image will reset the generated image pixels and the process of pixel search needs to be done again. That makes the process very very slow and does not gurantee good results. Due these time and compute constraints, it cannot be implemented in production.


## The Solution: 
The solution is to generalize the approach, using something like a neural net that learns to apply a specific type of style on any input image. Although this approach is also not very good but it is much better than the previous one.

**Advantages**:
* Much faster than the traditional approach
* requires us to train the model only once per style

**Disadvantages**:
* Each style requires its own weights for the model which means it requires a lot of space to save weights for each type of style.

## Fast Neural Style Transfer
<p align="center">
  <img src="https://www.fritz.ai/images/fast_style_transfer_arch.jpg" width="500">
  <br>
  <em>Fig 5. The Transfer Network</em>
</p>

Training a style transfer model requires two networks: a pre-trained feature extractor and a transfer network. The pre-trained feature extractor is used to avoid having to us paired training data. It’s usefulness arises from the curious tendency for individual layers of deep convolutional neural networks trained for image classification to specialize in understanding specific features of an image.

The pre-trained model enables us to compare the content and style of two images, but it doesn't actually help us create the stylized image. That’s the job of a second neural network, which we’ll call the transfer network. The transfer network is an image translation network that takes one image as input and outputs another image. Transfer networks typically have an encode-decoder architecture.

At the beginning of training, one or more style images are run through the pre-trained feature extractor, and the outputs at various style layers are saved for later comparison. Content images are then fed into the system. Each content image passes through the pre-trained feature extractor, where outputs at various content layers are saved. The content image then passes through the transfer network, which outputs a stylized image. The stylized image is also run through the feature extractor, and outputs at both the content and style layers are saved.

The quality of the stylized image is defined by a custom loss function that has terms for both content and style. The extracted content features of the stylized image are compared to the original content image, while the extracted style features are compared to those from the reference style image(s). After each step, only the transfer network is updated. The weights of the pre-trained feature extractor remain fixed throughout. By weighting the different terms of the loss function, we can train models to produce output images with lighter or heavier stylization. 

## Requirements:
1. Python == 3.7.6
2. Torch == 1.5.1
3. Torchvision == 0.6.0a0+35d732a
4. Numpy == 1.18.1
5. PIL == 5.4.1
6. tqdm == 4.45.0
7. Matplotlib == 3.2.1
8. OpenCV == 4.2.0.34
9. CUDA Version == 10.1

## Installation and Usage:

Clone this repo:
```
git clone https://github.com/yash-choudhary/Neural-Style-Transfer.git
```

Install the dependencies
```
pip3 install -r requirements.txt
```

Just open the provided Fast Neural Style Transfer.ipynb in colab or your local GPU enabled machine. Run the **fast_trainer** function to train your custom model or use the provided pretrained model with the **test_image** function to generate results.
```
For reading purpose or more visually appealing results, you can just open the provided html file in a browser.
```

You can also see this notebook on [Kaggle.](https://www.kaggle.com/yashchoudhary/fast-neural-style-transfer)

## Experiments
I experimented with different layer formats and style and content weights and there are the results of each experiment.
| **Experiment Number** |  **1**  |  **2**  |  **3**  |  **4**  |  **5**  |
|-----------------------|:-------:|:-------:|:-------:|:-------:|:-------:|
| **batch_size**        |    4    |    4    |    4    |    8    |    4    |
| **epochs**            |    10   |    4    |    2    |    20   |    2    |
| **style_weight**      |   1e10  |  10e10  |  10e10  |  10e10  |  10e20  |
| **content_weight**    |   1e5   |   10e3  |   10e5  |   10e5  |   10e3  |
| **maxpool/avgpool**   | maxpool | maxpool | maxpool | avgpool | maxpool |
<br>
You can access the resuling images of each experiment in "experiments" folder of this repo.<br>

<p align="center">
  <img src="assets/grid.png" width="900">
  <br>
  <em>Fig 6. Experiment Results</em>
</p>

## Result
The 3 best outputs from my models are:

<p align="center">
  <img src="assets/train_loss.png" width="500">
  <br>
  <em>Fig 7. Training Loss</em>
</p>
<p align="center">
  <img src="Results/best_output1.jpg" width="500">
  <br>
  <em>Fig 8. Best Result 1 [More Weight to Style]</em>
</p>
<p align="center">
  <img src="Results/best_output2.jpg" width="500">
  <br>
  <em>Fig 9. Best Result 2 [Balanced Style and content]</em>
</p>
<p align="center">
  <img src="Results/best_output3.jpg" width="500">
  <br>
  <em>Fig 10. Best Result 3 [More Weight to Content]</em>
</p>

Please find detailed experiment results [here](https://drive.google.com/drive/folders/13jTfhQVB2qojOD3cb9EF7-Uy_afYUbDE?usp=sharing).

## Important Links
1. Train Dataset Link: http://images.cocodataset.org/zips/test2017.zip 
2. Style Image: https://github.com/myelinfoundry-2019/challenge/raw/master/picasso_selfportrait.jpg 
3. Content Image: https://github.com/myelinfoundry-2019/challenge/raw/master/japanese_garden.jpg 
4. Best Model: https://www.dropbox.com/s/7xvmmbn1bx94exz/best_model.pth?dl=1

## References:
1. [Style Transfer Guide](https://www.fritz.ai/style-transfer/)
2. [Breaking Down Leon Gatys’ Neural Style Transfer in PyTorch](https://towardsdatascience.com/breaking-down-leon-gatys-neural-style-transfer-in-pytorch-faf9f0eb79db)
3. [Intuitive Guide to Neural Style Transfer](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee)
4. [A Neural Algorithm of Artistic Style ByLeon A. Gatys, Alexander S. Ecker, Matthias Bethge](https://arxiv.org/abs/1508.06576)
5. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution by Justin Johnson, Alexandre Alahi, Li Fei-Fei](https://arxiv.org/abs/1603.08155)
6. [Neural Style Transfer on Real Time Video (With Full implementable code)](https://towardsdatascience.com/neural-style-transfer-on-real-time-video-with-full-implementable-code-ac2dbc0e9822)
7. ![Classic Neural Style Transfer](https://github.com/halahup/NeuralStyleTransfer)
8. ![Fast Neural Style Transfer using Lua](https://github.com/lengstrom/fast-style-transfer)
9. ![Fast Neural Style Transfer using Python](https://github.com/eriklindernoren/Fast-Neural-Style-Transfer)
