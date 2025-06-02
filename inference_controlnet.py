import json
import os
import gc
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
from diffusers.utils import load_image
import sa_handler
from tqdm import tqdm
import random  # 用于生成随机种子

# 定义输入和输出的主文件夹路径
# input_json = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/inference_image_paths_without_depth.json"  # 存储路径的 JSON 文件
input_json = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/inference_image_paths_without_depth_test_angle_00.json"  # 存储路径的 JSON 文件
output_main_folder = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/generated_images_520100_test_angle_00"

# 基础模型和组件路径
base_model_path = "/ssd/du_dataset/diffusers/examples/stable-diffusion-v1-5"
controlnet_path = "/ssd/du_dataset/diffusers/examples/controlnet/output_new/checkpoint-520100/controlnet"
unet_path = "/ssd/du_dataset/diffusers/examples/controlnet/output_new/checkpoint-520100/unet"

# 初始化 ControlNet 和 UNet 模型
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
custom_unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)

# 初始化管道并加载自定义的 UNet 模型
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, unet=custom_unet, torch_dtype=torch.float16, safety_checker=None
)

# 配置推理过程的优化
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

# StyleAlignedArgs 和 Handler 设置
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False, share_layer_norm=False,
                                      share_attention=True, adain_queries=True, adain_keys=True, adain_values=False)
handler = sa_handler.Handler(pipe)
handler.register(sa_args)

# 加载 JSON 文件中的图像路径信息
with open(input_json, "r") as f:
    folder_image_paths = json.load(f)

# 生成图像的参数
prompt = "A university building, realistic style, high-resolution, solid black background."
num_variants = 1  # 每张图像生成的方案数量
batch_size = 60 # 批量处理多少张
num_inference_steps = 100 # 设置推测步数

# 使用固定间隔的种子
seed_base = 42  # 基础种子值，每个变体加上固定间隔
seed_interval = 500  # 种子间隔

# Define the start and end folders
start_folder = "000"
end_folder = "009"
start_processing = False

# 遍历 JSON 文件中的每个子文件夹
for folder_name, image_paths in folder_image_paths.items():
    # Check if the current folder is the start folder or if processing has already started
    if folder_name == start_folder:
        start_processing = True
    if not start_processing:
        continue  # Skip folders until reaching the specified start folder

    # Stop processing if the current folder is beyond the end folder
    if folder_name > end_folder:
        break

    input_images = [load_image(img_path) for img_path in image_paths]  # 加载子文件夹中的所有图片
    if not input_images:
        continue  # 跳过空的子文件夹

    # 创建输出子文件夹路径
    folder_output = os.path.join(output_main_folder, folder_name)
    os.makedirs(folder_output, exist_ok=True)

    # 为当前子文件夹生成图像
    prompts = [prompt] * len(input_images)
    
    for variant_idx in range(num_variants):
        variant_folder = os.path.join(folder_output, f"variant_{variant_idx+1}")
        os.makedirs(variant_folder, exist_ok=True)

        generated_images = []
        # generator = torch.manual_seed(variant_idx)  # 每个方案使用不同的种子

        # 使用固定间隔的种子来控制生成多样性
        generator = torch.manual_seed(seed_base + variant_idx * seed_interval)

        # 分批次生成图像
        for i in range(0, len(input_images), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_images = input_images[i:i + batch_size]

            with torch.autocast("cuda"):
                batch_generated = pipe(
                    batch_prompts,
                    image=batch_images,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    controlnet_conditioning_scale=1.0,
                    guidance_scale=6.5
                ).images

            generated_images.extend(batch_generated)

            # 清理内存
            del batch_generated
            gc.collect()
            torch.cuda.empty_cache()

        # 保存生成的多视角图像到当前方案的文件夹中
        for idx, image in enumerate(generated_images):
            image.save(os.path.join(variant_folder, f"output_view_{idx+1}.png"))

print("多视角图像生成完成。")
