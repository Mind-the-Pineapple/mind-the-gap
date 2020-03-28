from pathlib import Path
from helper_functions import resample_brain_mask

# resample brain maps voxels
PROJECT_ROOT = Path.cwd()
brain_mask_path = str(PROJECT_ROOT / 'data'/ 'masks' /
                      'MNI152_T1_2mm_brain_mask_dil1.nii.gz')
save_path = str(PROJECT_ROOT / 'data' / 'masks' /
'MNI152_T1_1.5mm_brain_masked.nii.gz')
resample_brain_mask(brain_mask_path, save_path, save_mask=True)

# resample the cubed mask
cubed_mask_path = str(PROJECT_ROOT / 'data' / 'masks' / 'squared_mask_2mm.nii')
save_path_cube = str(PROJECT_ROOT / 'data' / 'masks' /
'squared_mask_1.5mm.nii')
resample_brain_mask(cubed_mask_path, save_path_cube, save_mask=True)
