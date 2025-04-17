import os
from PIL import Image

def clean_images(data_dir):
    valid_extensions = (".jpg", ".jpeg", ".png")
    for folder in ['benign', 'malignant']:
        folder_path = os.path.join(data_dir, folder)
        
        # Ensure we are checking files in the folder, not subfolders
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            if os.path.isdir(file_path):  # Skip directories like 'SOB'
                print(f"Skipping directory: {file_path}")
                continue
            
            try:
                if not filename.lower().endswith(valid_extensions):  # Check if the file is not a valid image
                    os.remove(file_path)
                else:
                    img = Image.open(file_path)
                    img.verify()  # Verify the image integrity
            except PermissionError:
                print(f"Permission denied for: {file_path}")
                continue  # Skip files with permission issues
            except Exception as e:
                print(f"Removing corrupted image: {file_path}")
                os.remove(file_path)

# Usage
dataset_path = r"C:\Users\SNEHA\OneDrive\Desktop\Project1\breast"
clean_images(dataset_path)
