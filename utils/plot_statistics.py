import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_class_counts_histogram(npy_file, output_file):
    # Load class counts from npy file
    class_counts = np.load(npy_file)
    print(f"Class counts: {class_counts}")
    
    # Define class labels
    class_labels = [f"Class {i}" for i in range(len(class_counts))]
    print(f"Class labels: {class_labels}")

    # remnve the first class
    class_labels = class_labels[1:]
    class_counts = class_counts[1:]

    for i in range(len(class_counts)):
        print(f"Class {i+1}: {class_counts[i]} pixels")

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, class_counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Pixel Count')
    plt.title('Class Counts Histogram')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file)
    plt.show()
    print(f"Histogram saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot class counts histogram from a npy file.')
    parser.add_argument('--npy_file', '-in',type=str, help='Path to the class_counts.npy file')
    parser.add_argument('--output_file', '-out', type=str, help='Directory to save the histogram image')
    args = parser.parse_args()
    
    npy_file = Path(args.npy_file)

    plot_class_counts_histogram(npy_file, args.output_file)
