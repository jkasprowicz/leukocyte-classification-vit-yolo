#Média das heads do último bloco
#Attention Rollout com média das heads por camada

import torch
import torch.nn.functional as F
import timm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "vit_fold_1_best_base.pth"
IMAGE_PATH = "/lapix/dataset-vit/test/0102_1_065_jpg.rf.ecf5edd738cdde0acc3393832cbb2136.jpg"
NUM_CLASSES = 14

# ===============================
# TRANSFORM (EXATAMENTE IGUAL AO VAL)
# ===============================
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===============================
# LOAD MODEL
# ===============================
model = timm.create_model(
    "vit_base_patch16_384",
    pretrained=False,
    num_classes=NUM_CLASSES
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ===============================
# ATTENTION STORAGE
# ===============================
attentions = []

def forward_hook(module, input, output):
    # output shape: [B, tokens, dim]
    # precisamos recalcular atenção manualmente

    x = input[0]  # [B, tokens, dim]
    B, N, C = x.shape

    qkv = module.qkv(x)
    qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * module.scale
    attn = attn.softmax(dim=-1)

    attentions.append(attn.detach())

for blk in model.blocks:
    blk.attn.register_forward_hook(forward_hook)


# ===============================
# LOAD IMAGE
# ===============================
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ===============================
# FORWARD PASS
# ===============================
with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(1).item()

print("Predicted class:", pred_class)
print("Num attention layers:", len(attentions))
print("Attention shape:", attentions[0].shape)
# ===============================
# ATTENTION ROLLOUT
# ===============================
# attentions: list of [B, heads, tokens, tokens]

attention_rollout = torch.eye(attentions[0].size(-1)).to(device)

for attn in attentions:
    attn_heads_fused = attn.mean(dim=1)  # average heads
    attn_heads_fused = attn_heads_fused.squeeze(0)

    # add residual connection
    attn_heads_fused = attn_heads_fused + torch.eye(attn_heads_fused.size(0)).to(device)
    attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)

    attention_rollout = torch.matmul(attn_heads_fused, attention_rollout)

# CLS token attention
mask = attention_rollout[0, 1:]  # remove CLS
grid_size = int(np.sqrt(mask.size(0)))

mask = mask.reshape(grid_size, grid_size)
mask = mask / mask.max()

# Upsample
mask = mask.unsqueeze(0).unsqueeze(0)
mask = F.interpolate(mask, size=(384, 384), mode="bilinear", align_corners=False)
mask = mask.squeeze().cpu().numpy()

# ===============================
# PLOT
# ===============================
plt.figure(figsize=(6,6))
plt.imshow(image.resize((384,384)))
plt.imshow(mask, cmap='jet', alpha=0.5)
plt.axis("off")
plt.title("ViT Attention Map")
plt.savefig("vit_attention_overlay.png", dpi=300)
plt.show()
