import os

def rename_images(folder_path, prefix="image"):
    # get all image files and sort them
    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ])

    for idx, filename in enumerate(files, start=1):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1]  # keep original extension
        new_name = f"{prefix}_{idx}{ext}"
        new_path = os.path.join(folder_path, new_name)

        os.rename(old_path, new_path)

    print(f"Renamed {len(files)} images in {folder_path}.")

# Example usage:
folder = "BgImg/Mountains"#"./Outputs/RenderedImages/airplane_grey/bg"
rename_images(folder, prefix="bg")
