import numpy as np
import matplotlib.pyplot as plt
import os

# Using a Raw String (r'') to handle the Windows backslashes and special characters
data_path = r'C:\Users\ayoub\Desktop\DL_PROJECT\PROJECT\Footprint_Classification_Project\just one more thing!\DinoTracker-main\DinoTracker-main\data'

images_file = os.path.join(data_path, 'images_compressed.npz')
names_file = os.path.join(data_path, 'names.npy')

def visualize_dino_data():
    print(f"Checking path: {data_path}")
    
    # 1. Load the compressed images
    if not os.path.exists(images_file):
        print(f"❌ Error: Cannot find {images_file}")
        return

    with np.load(images_file) as data:
        # Get the keys (e.g., 'images', 'X', etc.)
        keys = data.files
        print(f"✅ Success! Keys found in .npz: {keys}")
        
        # We'll use the first key found to get the images
        images = data[keys[0]]
        print(f"Dataset contains {images.shape[0]} images with resolution {images.shape[1:]}")

    # 2. Load the names
    if os.path.exists(names_file):
        names = np.load(names_file, allow_pickle=True)
        print(f"✅ Loaded {len(names)} names.")
    else:
        names = [f"Unknown_{i}" for i in range(len(images))]
        print("⚠️ names.npy not found, using generic IDs.")

    # 3. Plotting the Grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    fig.suptitle('DinoTracker: Sample Dinosaur Footprints', fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Display image (cmap='gray_r' flips it to black on white if needed)
            ax.imshow(images[i], cmap='gray')
            ax.set_title(names[i][:20], fontsize=8) # Truncate long names
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save a copy in your current folder just in case plt.show() fails
    plt.savefig('dino_check.png')
    print("✅ Preview saved to 'dino_check.png'")
    plt.show()

if __name__ == "__main__":
    visualize_dino_data()