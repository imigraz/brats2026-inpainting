import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class BrainImageGenerator(Dataset):
    def __init__(self, setup_file_path='dataset/setup/train.txt', train=True, test=False):
        self.train = train
        self.test = test

        with open(setup_file_path, 'r') as f:
            self.folder_paths = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, index):
        folder_path = self.folder_paths[index]
        sample_id = Path(folder_path).name

        t1_img_path = os.path.join(folder_path, f'{sample_id}-t1n.nii.gz')
        mask_unhealthy_path = os.path.join(folder_path, f'{sample_id}-mask-unhealthy.nii.gz')

        # Load image and mask
        sitk_image = sitk.ReadImage(t1_img_path)
        sitk_mask = sitk.ReadImage(mask_unhealthy_path)

        image_np = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        mask_np = sitk.GetArrayFromImage(sitk_mask).astype(np.float32)

        # Normalize the volume
        max_val = image_np.max()
        if max_val > 0:
            image_norm = (image_np / max_val * 2.0) - 1.0
        else:
            image_norm = image_np

        # Crop to (144, 208, 208)
        image_cropped = image_norm[5:149, 16:224, 16:224]
        mask_cropped = mask_np[5:149, 16:224, 16:224]

        # Select a random z slice
        z_idx = np.random.randint(1, image_cropped.shape[0] - 1)

        # Stack z-1, z, z+1 --> shape: (3, 208, 208)
        slice_image = image_cropped[z_idx - 1:z_idx + 2]
        image_tensor = torch.from_numpy(slice_image)

        slice_mask = mask_cropped[z_idx - 1:z_idx + 2]
        mask_unhealthy_tensor = torch.from_numpy(slice_mask)

        if self.train:
            return image_tensor, mask_unhealthy_tensor
        else:
            # Load extra masks for validation
            mask_healthy_path = os.path.join(folder_path, f'{sample_id}-mask-healthy.nii.gz')
            full_mask_path = os.path.join(folder_path, f'{sample_id}-mask.nii.gz')

            sitk_mask_healthy = sitk.ReadImage(mask_healthy_path)
            sitk_full_mask = sitk.ReadImage(full_mask_path)

            mask_h_np = sitk.GetArrayFromImage(sitk_mask_healthy).astype(np.float32)[5:149, 16:224, 16:224]
            mask_f_np = sitk.GetArrayFromImage(sitk_full_mask).astype(np.float32)[5:149, 16:224, 16:224]

            mask_healthy_tensor = torch.from_numpy(mask_h_np[z_idx - 1:z_idx + 2])
            full_mask_tensor = torch.from_numpy(mask_f_np[z_idx - 1:z_idx + 2])

            return image_tensor, mask_unhealthy_tensor, mask_healthy_tensor, full_mask_tensor
