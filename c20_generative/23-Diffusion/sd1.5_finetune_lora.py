# -*- coding: utf-8 -*-
# @Time    : 2025/10/29
# @Author  : CFuShn
# @Comments: SD1.5 LoRA fine-tuning with MassiveH5Dataset + Accelerate
# @Software: PyCharm

import h5py
import io
import numpy as np
import random
import torch

from PIL import Image
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Dataset
# ============================================================
class MassiveH5Dataset(Dataset):
    def __init__(self, h5_path, prompt_col="prompt", image_col="images",
        image_type=0, resize=512, transform=None, verbose=True):
        self.h5_path = h5_path
        self.image_col = image_col
        self.image_type = image_type
        self.resize = resize
        self.transform = transform
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            sample_prompts = random.sample(f[prompt_col][:].flatten().tolist(),
                                           20)
            print(sample_prompts)
            self.prompts = f[prompt_col][:]
            self.images_ref = f[image_col]

        self.length = len(self.prompts)
        if verbose:
            print(f"\nFinal dataset size: {self.length}")

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        hf = self._require_h5()
        # --- 图像读取 ---
        if self.image_type == 0:
            jpg_bytes = hf[self.image_col][idx]
            img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
        elif self.image_type == 1:
            arr = hf[self.image_col][idx]
            img = Image.fromarray((arr * 255).astype(
                np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8))
        else:
            raise ValueError(f"Unknown image_type: {self.image_type}")

        if self.resize:
            img = img.resize((self.resize, self.resize))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1))

        prompt = self.prompts[idx]
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        return {"image": img, "prompt": prompt}

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except:
                pass


def collate_fn(batch):
    imgs = torch.stack([x["image"] for x in batch])
    prompts = [x["prompt"] for x in batch]
    return {"pixel_values": imgs, "prompts": prompts}


# ============================================================
# Accelerator Setup
# ============================================================
accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

print(f"🚀 Using device: {device}")

# ============================================================
# Load Base Model
# ============================================================
base_model = "/home/shared/SD1.5"

pipe = StableDiffusionPipeline.from_pretrained(base_model,
                                               torch_dtype=torch.float16)
pipe.to(device)

# text_encoder 必须 float32 否则 LayerNorm 报错
pipe.text_encoder.to(dtype=torch.float32, device=device)

vae = pipe.vae
unet = pipe.unet
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler

# ============================================================
# Apply LoRA to UNet
# ============================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_v"],
    lora_dropout=0.05,
)
unet = get_peft_model(unet, lora_config)
unet.train()

# ============================================================
# Dataset & Dataloader
# ============================================================
dataset = MassiveH5Dataset(
    "/home/cy/datasets/facial/MixedFace/MixedFace_202510201043.h5")
dataloader = DataLoader(dataset, batch_size=2, shuffle=True,
                        collate_fn=collate_fn, num_workers=4)

# ============================================================
# Optimizer
# ============================================================
optimizer = optim.AdamW(unet.parameters(), lr=1e-4)

# ============================================================
# Prepare all with Accelerator
# ============================================================
unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
vae.to(device)
text_encoder.to(device)

for p in text_encoder.parameters():
    p.requires_grad = False

# ============================================================
# Training Loop
# ============================================================
for epoch in range(3):
    progress_bar = tqdm(dataloader,
                        disable=not accelerator.is_local_main_process)
    for batch in progress_bar:
        imgs = batch["pixel_values"].to(device, dtype=torch.float16)
        prompts = batch["prompts"]

        tokens = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.to(device)
        attn_mask = tokens.attention_mask.to(device)

        # 编码文本
        with torch.no_grad():
            text_embeds = text_encoder(input_ids, attention_mask=attn_mask)[0]
            text_embeds = text_embeds.to(dtype=torch.float16)  # 统一fp16,保证类型一致

        # 编码图像到latent空间
        with torch.no_grad():
            latents = vae.encode(
                imgs).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(torch.float16)  # 统一fp16,保证类型一致

        # 加噪音
        # noise = torch.randn_like(latents)
        noise = torch.randn_like(latents, dtype=torch.float16)  # 统一fp16,保证类型一致
        timesteps = torch.randint(0, scheduler.num_train_timesteps,
                                  (latents.size(0),), device=device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # 预测噪声
        noise_pred = unet(noisy_latents, timesteps,
                          encoder_hidden_states=text_embeds).sample

        # loss = nn.MSELoss()(noise_pred, noise)
        loss = nn.MSELoss()(noise_pred.float(), noise.float())  # 确保统一f32精度
        accelerator.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet.save_pretrained(f"/home/shared/SD1.5_finetune_lora_epoch{epoch}",
                             safe_serialization=True)
        print(f"✅ Epoch {epoch} saved!")

accelerator.print("🎯 All epochs finished. LoRA fine-tuning complete.")
