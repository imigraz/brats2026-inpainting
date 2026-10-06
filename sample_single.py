import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.backends.cuda
import torch.backends.cudnn
from tqdm import tqdm

from generative_model.trainer_fm import Flow
from generative_model.unet import create_model


OUTPUT_DIR = Path('output')


def get_args():
    parser = argparse.ArgumentParser(description="Zero-Shot Brain MRI Single Reconstruction")
    parser.add_argument("--input_path", type=str, required=True, help="Path to *-t1n-voided.nii.gz")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to *-mask.nii.gz")
    parser.add_argument("--checkpoint", type=str, default="models/model-10k.pt", help="Path to model weights")
    parser.add_argument("--ode_steps", type=int, default=32, help="Number of ODE steps")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_model(checkpoint_path, ode_steps, device):
    unet_model = create_model(image_size=208, num_channels=128, num_res_blocks=2, in_channels=3, out_channels=3,
                              use_scale_shift_norm=True, dims=2).cuda()

    flow_model = Flow(unet_model, image_size=208, input_channels=3, ode_steps=ode_steps).cuda()

    flow_model.load_state_dict(torch.load(checkpoint_path, map_location=device)['ema'])
    flow_model.eval()

    return flow_model


def get_test_image(img_path, mask_path):
    img_sitk = sitk.ReadImage(img_path)
    mask_sitk = sitk.ReadImage(mask_path)

    img_np = sitk.GetArrayFromImage(img_sitk).astype(np.float32)[5:149, 16:224, 16:224]
    mask_np = sitk.GetArrayFromImage(mask_sitk).astype(np.float32)[5:149, 16:224, 16:224]

    max_val = img_np.max()
    img_norm = (img_np / max_val * 2.0) - 1.0 if max_val > 0 else img_np

    return torch.from_numpy(img_norm), mask_np, max_val, img_sitk


def inpaint(args, model):
    img_voided, mask_full_np, max_val, img_voided_sitk = get_test_image(args.input_path, args.mask_path)

    base_vol_gpu = img_voided.to(args.device)
    mask_full_gpu = torch.from_numpy(mask_full_np).to(args.device)
    num_slices = base_vol_gpu.shape[0]

    reconstructed_vol = base_vol_gpu.clone()

    # Slices to inpaint
    masked_slice_indices = [
        z for z in range(1, num_slices - 1) if mask_full_gpu[z].sum() > 0
    ]

    pbar = tqdm(
        masked_slice_indices,
        desc=f"Inpainting {Path(args.input_path).name}",
        unit="slice",
    )

    for z in pbar:
        stack_tensor = reconstructed_vol[z - 1: z + 2].unsqueeze(0)

        mask_stack = mask_full_gpu[z - 1: z + 2].clone()
        if z > 1:
            mask_stack[0] = 0
        restora_mask = 1.0 - mask_stack.unsqueeze(0)

        with torch.no_grad():
            pred_stack = model.sample_restora_flow_mask_guided(
                input_img=stack_tensor,
                mask=restora_mask,
                ode_steps=args.ode_steps,
                correction_steps=1
            )

        pred_center_slice = pred_stack[0, 1, :, :]
        reconstructed_vol[z] = torch.where(mask_full_gpu[z] == 1, pred_center_slice, reconstructed_vol[z])

    reconstructed_vol = reconstructed_vol.cpu().numpy()

    # Post-processing
    if max_val > 0:
        reconstructed_vol = ((reconstructed_vol + 1.0) / 2.0) * max_val
        reconstructed_vol = np.clip(reconstructed_vol, a_min=0, a_max=max_val)

    pad_width = ((5, 6), (16, 16), (16, 16))
    padded_vol = np.pad(reconstructed_vol, pad_width, mode='constant', constant_values=0)

    size, spacing, origin, direction = (img_voided_sitk.GetSize(), img_voided_sitk.GetSpacing(),
                                        img_voided_sitk.GetOrigin(), img_voided_sitk.GetDirection())
    out_img = sitk.GetImageFromArray(padded_vol)
    out_img.SetSpacing(spacing)
    out_img.SetOrigin(origin)
    out_img.SetDirection(direction)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sitk.WriteImage(out_img, os.path.join(OUTPUT_DIR, Path(args.input_path).name))


if __name__ == "__main__":
    args = get_args()
    flow_model = load_model(args.checkpoint, args.ode_steps, args.device)
    inpaint(args, flow_model)
