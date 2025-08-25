import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import argparse
import math

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate GIF from LB / Ref / UB Images")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Path to folder containing lb_*.png, ref_*.png, ub_*.png images",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default=None,
        help="Folder to save the output GIF",
    )
    parser.add_argument(
        "--domain_type",
        type=str,
        required=True,
        help="Type of domain ('round' for angle display, else just index)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Height of images in GIF (width adjusted to keep aspect ratio)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    domain_type = args.domain_type
    img_h = args.height

    if args.folder is None:
        folder =  "Outputs/AbstractImages/"+domain_type
    else:
        folder = args.folder
    if args.save_folder is None:
        save_folder = "Outputs/AbstractGif/"+domain_type
    else:
        save_folder = args.save_folder

    # Create save folder if it does not exist
    os.makedirs(save_folder, exist_ok=True)

    # Load images
    images_dict = {}
    for fname in os.listdir(folder):
        if fname.endswith(".png") and (fname.startswith("lb_") or fname.startswith("ub_") or fname.startswith("ref_")):
            images_dict[fname] = Image.open(os.path.join(folder, fname))

    # Extract max index from lb files
    indices = [int(f.split('_')[1].split('.')[0]) for f in images_dict if f.startswith("lb_")]
    max_index = max(indices) if indices else 0
    total_num = max_index + 1

    # Generate frames for GIF
    frames = []
    for num in range(total_num):
        row_images = []
        for prefix in ["lb_", "ref_", "ub_"]:
            key = f"{prefix}{num}.png"
            if key in images_dict:
                img = images_dict[key]
                w = int(img.width * (img_h / img.height))
                img_resized = img.resize((w, img_h), Image.Resampling.LANCZOS)
                row_images.append(img_resized)
            else:
                row_images.append(Image.new("RGB", (img_h, img_h), (255, 255, 255)))

        # Concatenate images horizontally
        total_w = sum(im.width for im in row_images)
        combined_img = Image.new("RGB", (total_w, img_h))
        x_offset = 0
        for im in row_images:
            combined_img.paste(im, (x_offset, 0))
            x_offset += im.width

        # Optional: draw degree display for "round"
        try:
            

            # Get the default PIL font directory
            font_path = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans.ttf")
            font = ImageFont.truetype(font_path, size=20)
            # font = ImageFont.load_default()
            draw = ImageDraw.Draw(combined_img)

            if domain_type == "round":
                degrees = (2 * math.pi / total_num) * num * 180 / math.pi
                draw.text((10, 10), f"{degrees:.1f}°", fill=(255, 0, 0), font=font)
            elif domain_type == "z":
                val = (num / total_num -1/2) * 15
                draw.text((10, 10), f"{val:.1f}ft", fill=(255, 0, 0), font=font)
            elif domain_type == "x":
                val = -(num / total_num -1/2) * 10+90
                draw.text((10, 10), f"{val:.1f}ft", fill=(255, 0, 0), font=font)
            elif domain_type == "y":
                val = (num / total_num -1/2) * 60+30
                draw.text((10, 10), f"{val:.1f}ft", fill=(255, 0, 0), font=font)
            elif domain_type == "yaw":
                degrees = (num / total_num -1/2) * 60 
                draw.text((10, 10), f"{degrees:.1f}°", fill=(255, 0, 0), font=font)
        except:
            font = None
            #draw.text((10, 10), f"{degrees:.1f}°", fill=(255, 0, 0), font=font)

        frames.append(combined_img)

    # Save as GIF
    save_path = os.path.join(save_folder, "lb_ref_ub_animation.gif")
    frames[0].save(save_path, save_all=True, append_images=frames[1::20], duration=5000/total_num, loop=0)
    print(f"GIF saved to {save_path}")


if __name__ == "__main__":
    main()
