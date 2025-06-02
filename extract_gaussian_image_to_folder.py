import os
import shutil

# Define source and destination folders
source_folder = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/gaussian_images_520100"
destination_folder = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/gaussian_images_520100_test_extracted"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Traverse all subfolders and files in the source folder
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png"):  # Only process PNG files
            # Construct the full path to the file
            full_path = os.path.join(root, file)
            
            # Extract relative path to create new filename
            relative_path = os.path.relpath(full_path, source_folder)
            parts = relative_path.split(os.sep)
            
            # Create new filename by combining parent folder and filename
            new_filename = f"{parts[0]}_{file}"
            new_file_path = os.path.join(destination_folder, new_filename)
            
            # Copy the file to the destination folder with the new name
            shutil.copy2(full_path, new_file_path)

print(f"Images have been successfully renamed and copied to {destination_folder}")
