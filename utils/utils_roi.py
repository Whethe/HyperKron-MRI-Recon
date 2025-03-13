import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.utils_base import mkdir


def process_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    m_list = []
    for ann in sorted_anns:
        m = ann['segmentation']
        m_list.append(m)
        print(m.shape)
    ms = np.stack(m_list, axis=0)

    return ms

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
def get_mask_sam_auto(img, n_mask=None, is_debug=False, device="cuda"):

    if len(img.shape) == 2:  # (h, w)
        pass
    elif len(img.shape) == 3:  # (c, h, w) --> (h, w)
        img = img[0, ...]
    else:
        raise ValueError("Image shape should be 2 or 3.")

    img = (img * 255).astype(np.uint8)  # (h, w) 0-1 --> (h, w) 0-255
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # (h, w) 0-255 --> (h, w, 3) 0-255

    # sam = sam_model_registry["vit_b"](checkpoint="weight/sam_vit_b_01ec64.pth").to(device)
    sam = sam_model_registry["vit_h"](checkpoint="weight/sam_vit_h_4b8939.pth").to(device)

    mask_generator = SamAutomaticMaskGenerator(model=sam)
    masks_pack = mask_generator.generate(img)

    if n_mask is not None:
        masks_pack = masks_pack[:n_mask]

    if is_debug:
        from utils.utils_sam import show_anns
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks_pack)
        mkdir(os.path.join('tmp', 'roi'))
        plt.savefig(os.path.join('tmp', 'roi', f'mask_sam.png'))
        plt.close()

    masks = process_anns(masks_pack)

    return masks


