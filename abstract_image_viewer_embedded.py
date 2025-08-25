import os
import base64
from io import BytesIO
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Scale, HORIZONTAL
import argparse
import math

# Parse command line argument
parser = argparse.ArgumentParser(description="LB / Ref / UB Image Viewer")
parser.add_argument(
    "--folder",
    type=str,
    default=None,
    help="Path to folder containing lb_*.png, ref_*.png, ub_*.png images",
)
parser.add_argument(
    "--domain_type",
    type=str,
    required=True,
    help="Type of domain ('round' for angle display, else just index)"
)
args = parser.parse_args()
domain_type = args.domain_type
if args.folder is None:
    folder = "Outputs/AbstractImages/"+domain_type
else:
    folder = args.folder

# Embed all images into a dictionary as base64 strings
images = {}
for fname in os.listdir(folder):
    if fname.endswith(".png") and (fname.startswith("lb_") or fname.startswith("ub_") or fname.startswith("ref_")):
        with open(os.path.join(folder, fname), "rb") as f:
            images[fname] = base64.b64encode(f.read()).decode("utf-8")

# Extract max index from lb files
indices = [int(f.split('_')[1].split('.')[0]) for f in images if f.startswith("lb_")]
max_index = max(indices) if indices else 0
total_num = max_index + 1

# Tkinter window
root = tk.Tk()
root.title("LB / Ref / UB Image Viewer")

# Titles
tk.Label(root, text="Lower Bound", font=("Arial", 14)).grid(row=0, column=0)
tk.Label(root, text="Ref Image", font=("Arial", 14)).grid(row=0, column=1)
tk.Label(root, text="Upper Bound", font=("Arial", 14)).grid(row=0, column=2)

# Image labels
lb_label = tk.Label(root)
lb_label.grid(row=1, column=0)
ref_label = tk.Label(root)
ref_label.grid(row=1, column=1)
ub_label = tk.Label(root)
ub_label.grid(row=1, column=2)

# Label for slider value display
value_label = tk.Label(root, text="", font=("Arial", 12))
value_label.grid(row=3, column=0, columnspan=3)


# Helper to resize image to height 300 while keeping aspect ratio
def resize_image_keep_ratio(img, target_height=300):
    h, w = img.size[1], img.size[0]
    new_w = int(target_height * w / h)
    return img.resize((new_w, target_height), Image.LANCZOS)

# Function to update images based on slider
def update_image(num):
    num = int(num)
    # Update slider value display
    if domain_type == "round":
        degrees = num / total_num * 360
        value_label.config(text=f"{degrees:.1f}°")
    elif domain_type == "z":
        val = -(num / total_num -1/2) * 15
        value_label.config(text=f"{val:.1f}ft")
    elif domain_type == "x":
        val = -(num / total_num -1/2)* 10+90
        value_label.config(text=f"{val:.1f}ft")
    elif domain_type == "y":
        val = (num / total_num -1/2) * 60+30
        value_label.config(text=f"{val:.1f}ft")
    elif domain_type == "yaw":
        degrees = (num / total_num -1/2) * 60 
        value_label.config(text=f"{degrees:.1f}ft")
    else:
        value_label.config(text=str(num))

    # Update images
    for prefix, label in [("lb_", lb_label), ("ref_", ref_label), ("ub_", ub_label)]:
        key = f"{prefix}{num}.png"
        if key in images:
            img_data = base64.b64decode(images[key])
            img = Image.open(BytesIO(img_data))
            # img = img.resize((300, 300))  # adjust size if needed
            # Resize: height = 300, width adjusted to maintain aspect ratio
            h = 300
            w = int(img.width * (h / img.height))
            img = img.resize((w, h), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            label.config(image=photo)
            label.image = photo

# Slider
slider = Scale(root, from_=0, to=max_index, orient=HORIZONTAL, command=update_image, length=900)
slider.grid(row=2, column=0, columnspan=3)

# Initial display
update_image(0)

root.mainloop()
