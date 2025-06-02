import os
import json

# 定义主文件夹路径
# main_folder = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_30"
main_folder = "/ssd/du_dataset/mvdfusion/my_dataset_processed_blender_whiteblock_512_60_00_test"

# 初始化用于存储路径信息的字典
folder_image_paths = {}

# 遍历主文件夹下的所有子文件夹和文件
for root, dirs, files in os.walk(main_folder):
    # 只处理直接的子文件夹
    if root == main_folder:
        for subdir in dirs:
            subdir_path = os.path.join(main_folder, subdir)
            # 按文件名顺序收集不含 'depth' 的图片路径
            image_files = sorted(
                [file for file in os.listdir(subdir_path) if file.endswith(".png") and "depth" not in file]
            )
            # 将图片路径加入字典，按子文件夹名称
            folder_image_paths[subdir] = [os.path.join(subdir_path, file) for file in image_files]

# 按子文件夹名称排序（从小到大）
sorted_folder_image_paths = dict(sorted(folder_image_paths.items(), key=lambda item: int(item[0])))

# 定义保存路径的 JSON 文件名
json_file = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/inference_image_paths_without_depth_test_angle_00.json"

# 将路径信息保存到 JSON 文件中
with open(json_file, "w") as f:
    json.dump(sorted_folder_image_paths, f, indent=4)

print(f"不带 'depth' 的图片路径信息已按子文件夹名称顺序保存到 {json_file}")
