import os
import matplotlib.pyplot as plt

def plot_eda(data_dir):
    labels = ['Benign', 'Malignant']
    counts = [len(os.listdir(os.path.join(data_dir, 'benign'))), len(os.listdir(os.path.join(data_dir, 'malignant')))]

    plt.bar(labels, counts)
    plt.title('Dataset Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.show()

# Usage
dataset_path = r"C:\Users\SNEHA\OneDrive\Desktop\Project1\breast"
plot_eda(dataset_path)
