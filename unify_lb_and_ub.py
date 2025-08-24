import os
import numpy as np
from PIL import Image

def compute_and_save_abstract_images(save_folder_full):
    # Lists to hold loaded images
    lb_images = []
    ub_images = []

    # Load images
    for fname in os.listdir(save_folder_full):
        if fname.startswith("lb_") and fname.endswith(".png"):
            img = np.array(Image.open(os.path.join(save_folder_full, fname)), dtype=np.float32) / 255.0
            lb_images.append(img)
        elif fname.startswith("ub_") and fname.endswith(".png"):
            img = np.array(Image.open(os.path.join(save_folder_full, fname)), dtype=np.float32) / 255.0
            ub_images.append(img)

    # Check if images are found
    if lb_images:
        # Stack and compute pixel-wise minimum (unified lower bound)
        lb_stack = np.stack(lb_images, axis=0)
        unified_lb = np.min(lb_stack, axis=0)
        unified_lb_img = (unified_lb.clip(0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(unified_lb_img).save(os.path.join(save_folder_full, "unified_lb.png"))
        print("Unified lower bound saved as unified_lb.png")
    else:
        print("No lb images found.")

    if ub_images:
        # Stack and compute pixel-wise maximum (unified upper bound)
        ub_stack = np.stack(ub_images, axis=0)
        unified_ub = np.max(ub_stack, axis=0)
        unified_ub_img = (unified_ub.clip(0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(unified_ub_img).save(os.path.join(save_folder_full, "unified_ub.png"))
        print("Unified upper bound saved as unified_ub.png")
    else:
        print("No ub images found.")

if __name__ == '__main__':
    # Folder where images are saved
    save_folder_full = "./AbstractImages/output_y"
    compute_and_save_abstract_images(save_folder_full)