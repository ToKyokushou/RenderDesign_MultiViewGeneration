import json
import re

# 排序函数，按物体 ID 和视角编号对图像进行排序
def sort_views_by_object_and_angle(data):
    def extract_object_and_view_numbers(entry):
        match_object = re.search(r'/(\d+)/', entry["conditioning_image"])  # 提取物体 ID
        match_view = re.search(r'_view_(\d+)', entry["conditioning_image"])  # 提取视角编号
        object_id = int(match_object.group(1)) if match_object else -1
        view_number = int(match_view.group(1)) if match_view else -1
        return (object_id, view_number)
    return sorted(data, key=extract_object_and_view_numbers)

# 加载 JSON 数据
# with open('/ssd/du_dataset/diffusers/examples/controlnet/dataset/controlnet_image_paths_512_200_new.json', 'r') as f:
with open('/ssd/du_dataset/diffusers/examples/controlnet/dataset/controlnet_image_paths_512_200_new_angle_00.json', 'r') as f:
    data = json.load(f)

# 对 JSON 数据进行排序
sorted_data = sort_views_by_object_and_angle(data)

# 保存到新的 JSON 文件
# with open('/ssd/du_dataset/diffusers/examples/controlnet/dataset_sorted/controlnet_image_paths_512_200_new.json', 'w') as f:
with open('/ssd/du_dataset/diffusers/examples/controlnet/dataset_sorted_angle_00/controlnet_image_paths_512_200_new_angle_00.json', 'w') as f:
    json.dump(sorted_data, f, indent=4)

print("数据已按物体 ID 和视角排序并保存到新的 JSON 文件中。")
