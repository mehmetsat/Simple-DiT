# ReiMei: Image Diffusion Transformer

> **Note:** This code is from the winning solution of the Alphabet Dataset Challenge by [SwayStar123](https://github.com/SwayStar123). Original source: [alphabet-dataset-challenge discussion](https://github.com/fal-ai-community/alphabet-dataset/discussions/1)

ReiMei is an advanced image diffusion model built on the transformer architecture, designed for high-quality image generation with particular focus on shape and letter generation.

## 🔍 Overview

ReiMei combines elements from Diffusion Transformers (DiT) and Mixture of Experts (MoE) with a novel double-stream architecture (MM-DiT) to efficiently process and generate images. The model can generate images based on parametric shape descriptions, making it especially suitable for synthetic data generation and controlled image outputs.

## 🌟 Features

- **Multi-Modal Diffusion Transformer (MM-DiT)** architecture that efficiently processes both image and parameter data
- **Token Mixer** for enhanced cross-domain information exchange
- **Rotation Positional Encoding (RoPE)** for improved spatial awareness
- **Support for gradient filtering** to stabilize training
- **Flexible configuration** with support for various architectural choices:
  - Mixture of Experts (MoE)
  - Expert Choice (EC) routing
  - Shared attention projections and modulation layers

## 📋 Requirements

```
torch
einops
accelerate
tqdm
torchvision
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/yourusername/reimei.git
cd reimei
pip install -r requirements.txt
```

### Using the Model

```python
from reimei import ReiMei, ReiMeiParameters

# Configure model parameters
params = ReiMeiParameters(
    use_mmdit=True,
    use_ec=True,
    shared_mod=True,
    shared_attn_projs=True,
    channels=3,
    patch_size=(4, 4),
    embed_dim=768,
    num_layers=4,
    num_heads=6,  # embed_dim // 128
    num_experts=4,
    capacity_factor=2.0,
    shared_experts=1,
    dropout=0.1,
    token_mixer_layers=2,
    image_text_expert_ratio=2,
)

# Initialize model
model = ReiMei(params)

# For inference with trained model
noise = torch.randn(1, 3, 64, 64)
shape_params = torch.tensor([...])  # Shape parameters (27 values per image)
generated_image = model.sample(noise, shape_params, sample_steps=50, cfg=3.0)
```

## 🏋️ Training

The model is trained to predict velocity fields in a diffusion process. The training process:

1. Takes input images and their corresponding shape parameters
2. Applies noise to the images
3. Trains the model to predict the velocity vector that transforms the noisy image back to the original

```python
# Basic training loop example
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=100000)

for i in range(100000):
    for batch in train_dataloader:
        images, image_params = batch
        
        # Normalize images to [-1, 1]
        images = ((images - 0.5) * 2.0).to(device, dtype=torch.bfloat16)
        image_params = image_params.to(device, dtype=torch.bfloat16)
        
        # Sample noise
        z = torch.randn_like(images)
        t = torch.ones((images.shape[0],), device=device)
        
        # Forward pass
        vtheta = model(z, t, image_params)
        
        # Calculate velocity target
        v = z - images
        
        # Loss calculation
        loss = ((v - vtheta) ** 2).mean()
        
        # Backward pass with gradient filtering
        optimizer.zero_grad()
        loss.backward()
        grads = gradfilter_ema(model, grads=grads)
        optimizer.step()
        scheduler.step()
```

## 🧪 Sampling

Once trained, the model can generate images through an iterative denoising process:

```python
# Sample images
with torch.no_grad():
    model.eval()
    noise = torch.randn(16, 3, 64, 64).to(device)
    params = example_params[:16]
    
    # Generate images with 50 denoising steps and guidance scale of 5.0
    generated_images = model.sample(noise, params, sample_steps=50, cfg=5.0)
```

## 🧠 Model Architecture

ReiMei consists of several key components:

1. **Parameter Embedder**: Embeds shape parameters into the model's latent space
2. **Image Embedder**: Projects image patches into embeddings
3. **Token Mixer** (optional): Enhances cross-domain information flow
4. **Transformer Backbone**: Main processing component with multiple layers of attention
5. **Output Layer**: Projects embeddings back to image space

## 📊 Parameters

The model configuration is managed through the `ReiMeiParameters` dataclass:

- `use_mmdit`: Enable Multi-Modal Diffusion Transformer architecture
- `use_ec`: Use Expert Choice routing for MoE
- `use_moe`: Enable Mixture of Experts
- `shared_mod`: Share modulation layers across blocks
- `channels`: Number of image channels
- `patch_size`: Size of image patches
- `embed_dim`: Embedding dimension
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `num_experts`: Number of experts in MoE
- `token_mixer_layers`: Number of token mixer layers
- And more...

## 📝 Citations

The ReiMei model builds upon several important works in the field:

```
@article{peebles2023scalable,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  journal={arXiv preprint arXiv:2212.09748},
  year={2023}
}

@article{dai2022mixture,
  title={The Mixture-of-Experts Transformer: Combining Better Routing with Expert Selection Probabilities},
  author={Dai, Wenhao and others},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}
```




## 👥 Contributors

- [SwayStar123](https://github.com/SwayStar123) - Original author and winner of the Alphabet Dataset Challenge
- This implementation is based on the winning solution from the [Alphabet Dataset Challenge](https://github.com/fal-ai-community/alphabet-dataset/discussions/1)