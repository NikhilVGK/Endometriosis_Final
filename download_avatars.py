import os
import requests
from PIL import Image
from io import BytesIO

# Create directory if it doesn't exist
avatars_dir = os.path.join("static", "avatars")
os.makedirs(avatars_dir, exist_ok=True)

# 3D cartoon avatar URLs - using direct Image URLs
avatar_urls = [
    # Character avatars that are similar to the uploaded image style
    "https://img.freepik.com/free-vector/cute-boy-cartoon-character_1308-133791.jpg",
    "https://img.freepik.com/free-vector/cute-girl-cartoon-character_1308-133792.jpg",
    "https://img.freepik.com/free-vector/cute-boy-with-blue-shirt-cartoon-character_1308-133755.jpg",
    "https://img.freepik.com/free-vector/cute-girl-pink-shirt-cartoon-character_1308-133761.jpg",
    "https://img.freepik.com/free-vector/cute-boy-standing-position-cartoon-character_1308-133756.jpg",
    "https://img.freepik.com/free-vector/cute-girl-standing-position-cartoon-character_1308-133762.jpg",
    "https://img.freepik.com/free-vector/cute-boy-waving-hand-cartoon-character_1308-133754.jpg",
    "https://img.freepik.com/free-vector/cute-girl-waving-hand-cartoon-character_1308-133760.jpg",
    "https://img.freepik.com/free-vector/cute-schoolboy-cartoon-character_1308-133767.jpg",
    "https://img.freepik.com/free-vector/cute-schoolgirl-cartoon-character_1308-133766.jpg",
    "https://img.freepik.com/free-vector/cartoon-style-boy-character-design_1308-134020.jpg",
    "https://img.freepik.com/free-vector/cartoon-style-girl-character-design_1308-134021.jpg",
    "https://img.freepik.com/free-vector/cute-boy-cartoon-character-different-poses_1308-133793.jpg",
    "https://img.freepik.com/free-vector/cute-girl-cartoon-character-different-poses_1308-133794.jpg",
    "https://img.freepik.com/free-vector/boy-character-with-happy-face_1308-90506.jpg",
    "https://img.freepik.com/free-vector/girl-character-with-happy-face_1308-90507.jpg",
    "https://img.freepik.com/free-vector/cute-astronaut-cartoon-character_1308-133798.jpg",
    "https://img.freepik.com/free-vector/cute-astronaut-girl-cartoon-character_1308-133799.jpg",
    "https://img.freepik.com/free-vector/cute-superhero-boy-cartoon-character_1308-133800.jpg",
    "https://img.freepik.com/free-vector/cute-superhero-girl-cartoon-character_1308-133801.jpg",
    "https://img.freepik.com/free-vector/cute-pilot-boy-cartoon-character_1308-133802.jpg",
    "https://img.freepik.com/free-vector/cute-pilot-girl-cartoon-character_1308-133803.jpg",
    "https://img.freepik.com/free-vector/cute-soccer-player-boy-cartoon-character_1308-133804.jpg",
    "https://img.freepik.com/free-vector/cute-tennis-player-girl-cartoon-character_1308-133805.jpg",
    "https://img.freepik.com/free-vector/cute-painter-boy-cartoon-character_1308-133806.jpg",
    "https://img.freepik.com/free-vector/cute-painter-girl-cartoon-character_1308-133807.jpg",
    "https://img.freepik.com/free-vector/cute-chef-boy-cartoon-character_1308-133808.jpg",
    "https://img.freepik.com/free-vector/cute-chef-girl-cartoon-character_1308-133809.jpg",
    "https://img.freepik.com/free-vector/cute-doctor-boy-cartoon-character_1308-133810.jpg",
    "https://img.freepik.com/free-vector/cute-doctor-girl-cartoon-character_1308-133811.jpg",
    "https://img.freepik.com/free-vector/cute-scientist-boy-cartoon-character_1308-133812.jpg",
    "https://img.freepik.com/free-vector/cute-scientist-girl-cartoon-character_1308-133813.jpg"
]

# Download and save avatars
for i, url in enumerate(avatar_urls, 1):
    try:
        # Add headers to mimic browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            # Process and save image
            img = Image.open(BytesIO(response.content))
            
            # Convert to square image with white background
            size = max(img.size)
            new_img = Image.new('RGB', (size, size), color='white')
            new_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
            
            # Save as JPG
            output_path = os.path.join(avatars_dir, f"avatar_{i}.jpg")
            new_img.save(output_path, format='JPEG', quality=95)
            
            print(f"Downloaded avatar {i}/{len(avatar_urls)}: {output_path}")
        else:
            print(f"Failed to download avatar {i}: HTTP status {response.status_code}")
    except Exception as e:
        print(f"Error downloading avatar {i}: {str(e)}")

print("Avatar download complete!") 