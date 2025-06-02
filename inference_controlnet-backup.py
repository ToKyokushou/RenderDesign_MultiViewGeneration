from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
from diffusers.utils import load_image
import torch
from PIL import Image
import gc
import os
import sa_handler

input_folder = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/input_images/159"

# 定义保存生成图像的文件夹路径
output_folder = "/ssd/du_dataset/diffusers/examples/controlnet/inferences/generated_images/159"
os.makedirs(output_folder, exist_ok=True)  # 如果文件夹不存在，则创建

# 基础模型和组件路径
base_model_path = "/ssd/du_dataset/diffusers/examples/stable-diffusion-v1-5"  # 基础模型的路径
controlnet_path = "/ssd/du_dataset/diffusers/examples/controlnet/output_new/checkpoint-450000/controlnet"  # ControlNet的路径
unet_path = "/ssd/du_dataset/diffusers/examples/controlnet/output_new/checkpoint-450000/unet"  # 训练好的 UNet 模型路径

# 初始化 ControlNet 模型
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
custom_unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)

# 初始化管道并加载自定义的 UNet 模型
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, unet=custom_unet, torch_dtype=torch.float16, safety_checker=None
)

# 配置推理过程的优化
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_attention_slicing()  # 启用 attention slicing 以进一步降低显存占用
pipe.enable_model_cpu_offload()  # 启用模型 CPU offload

sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False,
                                  share_layer_norm=False,
                                  share_attention=True,
                                  adain_queries=True,
                                  adain_keys=True,
                                  adain_values=False,
                                 )
handler = sa_handler.Handler(pipe)
handler.register(sa_args, )

# 获取并加载多视角的调控图像
multi_view_images = []
for filename in sorted(os.listdir(input_folder)):  # 使用 sorted 保持加载顺序
    if filename.endswith(".png") or filename.endswith(".jpg"):  # 仅加载图像文件
        img_path = os.path.join(input_folder, filename)
        multi_view_images.append(load_image(img_path))

# 确保有图片加载，并设置提示词
if not multi_view_images:
    raise ValueError("No images found in input folder.")

prompt = "pale golden rod circle with old lace background"
prompts = [prompt] * len(multi_view_images)

# 分批次生成图像，减少显存占用
num_variants = 4  # 每张图像生成三种方案
batch_size = 6  # 设置批次大小

# 开始生成图像
for variant_idx in range(num_variants):
    variant_folder = os.path.join(output_folder, f"variant_{variant_idx+1}")
    os.makedirs(variant_folder, exist_ok=True)  # 为每个方案创建一个文件夹

    generated_images = []
    generator = torch.manual_seed(variant_idx)  # 每个方案使用不同的种子

    for i in range(0, len(multi_view_images), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_images = multi_view_images[i:i + batch_size]

        with torch.autocast("cuda"):
            batch_generated = pipe(
                batch_prompts,
                image=batch_images,
                num_inference_steps=100,
                generator=generator
            ).images

        generated_images.extend(batch_generated)

        # 清理内存
        del batch_generated
        gc.collect()
        torch.cuda.empty_cache()

    # 保存生成的多视角图像到当前方案的文件夹中
    for idx, image in enumerate(generated_images):
        image.save(os.path.join(variant_folder, f"output_view_{idx+1}.png"))

# 可选：使用wandb记录生成的多视角图像
# formatted_images = [wandb.Image(img, caption=f"View {i+1} - Variant {variant_idx+1}") for i, img in enumerate(generated_images)]
# wandb.log({f"multi_view_outputs_variant_{variant_idx+1}": formatted_images})
