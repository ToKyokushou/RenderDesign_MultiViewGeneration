import re
import json
from torch.utils.data import DataLoader, BatchSampler, Dataset
from PIL import Image
from torchvision import transforms

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 分组采样器，确保每个批次来自同一物体
class GroupBatchSampler(BatchSampler):
    def __init__(self, dataset, group_size, batch_size):
        self.dataset = dataset
        self.group_size = group_size  # 每个物体包含多少个视角
        self.batch_size = batch_size

    def __iter__(self):
        for idx in range(0, len(self.dataset), self.group_size):
            group_indices = list(range(idx, min(idx + self.group_size, len(self.dataset))))
            yield group_indices

    def __len__(self):
        return len(self.dataset) // self.group_size

# 加载数据集
json_file = '/ssd/du_dataset/diffusers/examples/controlnet/dataset_sorted/controlnet_image_paths_new_512_sorted.json'  # 预先排序好的 JSON 文件
custom_dataset = CustomDataset(json_file)

# 使用 GroupBatchSampler 生成 DataLoader，确保每个批次包含同一物体的所有视角
group_size = 60  # 每个物体有 60 个视角
batch_size = 1  # 每个批次包含一个物体的所有视角
sampler = GroupBatchSampler(custom_dataset, group_size=group_size, batch_size=batch_size)
train_dataloader = DataLoader(custom_dataset, batch_sampler=sampler, num_workers=2)

# 打印每个批次的数据
for batch_idx, batch in enumerate(train_dataloader):
    print(f"Batch {batch_idx}: {batch}")
    break
