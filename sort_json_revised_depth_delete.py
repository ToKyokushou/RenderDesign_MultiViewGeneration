import json

# 定义要替换的文本内容
new_text = "A university building, realistic style, high-resolution, solid black background."

# 读取 JSON 文件
with open("/ssd/du_dataset/diffusers/examples/controlnet/dataset_sorted/controlnet_image_paths_512_200_new.json", "r") as f:
    data = json.load(f)

# 遍历每个项目，修改 `text` 字段的值
for item in data:
    item["text"] = new_text

# 将修改后的数据保存回 JSON 文件
with open("/ssd/du_dataset/diffusers/examples/controlnet/dataset_sorted/controlnet_image_paths_512_200_new.json", "w") as f:
    json.dump(data, f, indent=4)

print("All text fields have been updated.")
