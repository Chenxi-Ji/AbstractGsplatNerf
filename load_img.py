import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the image
# -----------------------------
img_path = "./BgImg/mountain.jpg"  # change to your path
img = Image.open(img_path).convert("RGB")  # ensure 3 channels
img = img.resize((80,80), Image.LANCZOS) 

# -----------------------------
# 2. Convert to NumPy array
# -----------------------------
img_np = np.array(img, dtype=np.float32)  # shape: (H, W, 3)


# Normalize to [0,1]
img_np /= 255.0

# -----------------------------
# 3. Convert to PyTorch tensor
# -----------------------------
img_tensor = torch.from_numpy(img_np)  # shape: (H, W, 3)
print("Tensor shape:", img_tensor.shape)
print("Tensor dtype:", img_tensor.dtype)

# -----------------------------
# 4. Visualize using matplotlib
# -----------------------------
plt.imshow(img_tensor.numpy())
plt.axis('off')  # optional: hide axis
plt.title("Loaded Image")
plt.show()
