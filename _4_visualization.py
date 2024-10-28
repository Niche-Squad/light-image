from matplotlib import pyplot as plt
from pathlib import Path

# local imports
from model import CAE
from data import from_path_to_patches

ck = Path.cwd() / "logs" / "best.pt" 
img = Path.cwd() / "data" / "images" / "01-08" / "frame_00.jpg"

model = CAE.load_from_checkpoint(ck)
img, patches = from_path_to_patches(img)
patches, _ = model.merge_patch_to_batch(patches.unsqueeze(0))

model(patches[0].to("mps"))

# visualization
encoded = []
for i in range(16):
    tmp = model.encoded[i].cpu().detach().numpy()
    encoded += [tmp]

# 2 x 8
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    # rm axis tick
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.imshow(encoded[i], cmap="gray")
