from PIL import Image
import numpy as np

camvid_palette = [0, 0, 0, 255, 255, 255]  # 0 -> 黑色，1 -> 白色
zero_pad = 256 * 3 - len(camvid_palette)
for i in range(zero_pad):
    camvid_palette.append(0)

def camvid_colorize_mask(mask):
    # mask: numpy array，输入值预期为 [0, 255]，转换为 [0, 1] 映射颜色
    mask_mapped = (mask / 255).astype(np.uint8)  # 将 [0, 255] 转换为 [0, 1]
    new_mask = Image.fromarray(mask_mapped).convert('P')
    new_mask.putpalette(camvid_palette)  # 0 -> 黑色, 1 -> 白色
    return new_mask