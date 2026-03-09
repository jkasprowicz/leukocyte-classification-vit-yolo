import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# ATTENTION ROLLOUT
# ===============================

# attentions = lista já coletada via hooks
# cada item: [1, heads, tokens, tokens]

result = torch.eye(attentions[0].size(-1)).to(device)

for attn in attentions:
    
    # média das heads
    attn_heads_fused = attn.mean(dim=1)  # [1, tokens, tokens]
    
    # adiciona residual (I)
    attn_heads_fused = attn_heads_fused + torch.eye(attn_heads_fused.size(-1)).to(device)
    
    # normaliza
    attn_heads_fused = attn_heads_fused / attn_heads_fused.sum(dim=-1, keepdim=True)
    
    # multiplica acumulando
    result = torch.matmul(attn_heads_fused, result)

# CLS -> patches
mask = result[0, 0, 1:]  # remove CLS token

# reshape para grade espacial
num_patches = 24  # 384 / 16
mask = mask.reshape(1, 1, num_patches, num_patches)

# upsample para resolução original
mask = F.interpolate(mask, size=(384, 384), mode='bilinear', align_corners=False)

# normaliza 0-1
mask = mask.squeeze().detach().cpu().numpy()
mask = (mask - mask.min()) / (mask.max() - mask.min())

# ===============================
# PLOT
# ===============================

fig, ax = plt.subplots(1, 2, figsize=(8,4))

ax[0].imshow(image)
ax[0].axis("off")
ax[0].set_title("Original", fontsize=10)

ax[1].imshow(image)
ax[1].imshow(mask, cmap="jet", alpha=0.4)
ax[1].axis("off")
ax[1].set_title("ViT Attention Rollout", fontsize=10)

plt.tight_layout()
plt.savefig("vit_rollout_attention.png", dpi=300, bbox_inches="tight")
plt.show()
