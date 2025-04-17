import os

def verify_dataset(data_dir):
    for folder in ['benign', 'malignant']:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Missing folder: {folder_path}")
            return
        images = os.listdir(folder_path)
        print(f"Found {len(images)} images in '{folder}' folder.")
        for img in images[:5]:  # Display the first 5 images for verification
            print(f" - {img}")

# Usage
dataset_path = "C:\Users\SNEHA\OneDrive\Desktop\Project1\breast"
verify_dataset(dataset_path)
