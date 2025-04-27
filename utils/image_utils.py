from PIL import Image
import os

def verify_images(folder):
    """Check and remove corrupt images in a directory."""
    for img_file in os.listdir(folder):
        try:
            img_path = os.path.join(folder, img_file)
            with Image.open(img_path) as img:
                img.verify()
        except (IOError, SyntaxError) as e:
            print(f"Removing corrupt image: {img_path} | Error: {str(e)}")
            os.remove(img_path)

def check_all_image_folders():
    """Run verification on all image folders."""
    folders = ['data/images/positive', 'data/images/negative']
    for folder in folders:
        if os.path.exists(folder):
            print(f"Verifying {folder}...")
            verify_images(folder)