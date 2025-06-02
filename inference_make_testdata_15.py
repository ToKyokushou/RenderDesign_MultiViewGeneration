import os
import json

# Define the main folder path
main_folder = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_30"

# Initialize a dictionary to store path information
folder_image_paths = {}

# Traverse all subfolders and files in the main folder
for root, dirs, files in os.walk(main_folder):
    # Process only direct subfolders
    if root == main_folder:
        for subdir in dirs:
            subdir_path = os.path.join(main_folder, subdir)
            # Collect paths of images that do not contain 'depth' and sort them by filename
            image_files = sorted(
                [file for file in os.listdir(subdir_path) if file.endswith(".png") and "depth" not in file]
            )

            # Rearrange the images based on the rotated order every 15 images
            rearranged_files = []
            for i in range(15):
                rearranged_files.extend(image_files[i::15])  # Get every 15th image starting from index i

            # Add the paths to the dictionary under the subfolder name
            folder_image_paths[subdir] = [os.path.join(subdir_path, file) for file in rearranged_files]

# Sort by subfolder name in ascending order
sorted_folder_image_paths = dict(sorted(folder_image_paths.items(), key=lambda item: int(item[0])))

# Define the JSON file path to save
json_file = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/inference_image_paths_without_depth_15.json"

# Save the path information to a JSON file
with open(json_file, "w") as f:
    json.dump(sorted_folder_image_paths, f, indent=4)

print(f"Image paths without 'depth' have been saved in order to {json_file}")
