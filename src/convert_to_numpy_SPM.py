import os
import numpy as np

from nibabel.testing import data_path
import nibabel as nib


## 1
for i in range(0,623):
    PATHGM = 'gm/gm_part1/sub'+str(i)+'_gm'
    PATHWM = 'wm/wm_part1/sub'+str(i)+'_wm'
    try:
        img_gm = nib.load(PATHGM + '.nii.gz')
        img_wm = nib.load(PATHWM + '.nii.gz')

    except OSError:
        print("File " +PATHGM+ " might be corrupted")
        continue
    except:
        print("File "+PATHGM+ " doesn't seem to exist")
        continue
    img_gm = np.array(img_gm.dataobj)
    img_wm = np.array(img_wm.dataobj)
    np.save(PATHGM, img_gm)
    np.save(PATHWM, img_wm)
    print('Saved file '+ PATHGM)
    print('Saved file '+ PATHWM)


## 2
for i in range(625,1229):
    PATHGM = 'gm/gm_part2/sub'+str(i)+'_gm'
    PATHWM = 'wm/wm_part2/sub'+str(i)+'_wm'
    try:
        img_gm = nib.load(PATHGM + '.nii.gz')
        img_wm = nib.load(PATHWM + '.nii.gz')

    except OSError:
        print("File " +PATHGM+ " might be corrupted")
        continue
    except:
        print("File "+PATHGM+ " doesn't seem to exist")
        continue
    img_gm = np.array(img_gm.dataobj)
    img_wm = np.array(img_wm.dataobj)
    np.save(PATHGM, img_gm)
    np.save(PATHWM, img_wm)
    print('Saved file '+ PATHGM)
    print('Saved file '+ PATHWM)


## 3
for i in range(1231,1867):
    PATHGM = 'gm/gm_part3/sub'+str(i)+'_gm'
    PATHWM = 'wm/wm_part3/sub'+str(i)+'_wm'
    try:
        img_gm = nib.load(PATHGM + '.nii.gz')
        img_wm = nib.load(PATHWM + '.nii.gz')

    except OSError:
        print("File " +PATHGM+ " might be corrupted")
        continue
    except:
        print("File "+PATHGM+ " doesn't seem to exist")
        continue
    img_gm = np.array(img_gm.dataobj)
    img_wm = np.array(img_wm.dataobj)
    np.save(PATHGM, img_gm)
    np.save(PATHWM, img_wm)
    print('Saved file '+ PATHGM)
    print('Saved file '+ PATHWM)

## 4
for i in range(1868,2508):
    PATHGM = 'gm/gm_part4/sub'+str(i)+'_gm'
    PATHWM = 'wm/wm_part4/sub'+str(i)+'_wm'
    try:
        img_gm = nib.load(PATHGM + '.nii.gz')
        img_wm = nib.load(PATHWM + '.nii.gz')

    except OSError:
        print("File " +PATHGM+ " might be corrupted")
        continue
    except:
        print("File "+PATHGM+ " doesn't seem to exist")
        continue
    img_gm = np.array(img_gm.dataobj)
    img_wm = np.array(img_wm.dataobj)
    np.save(PATHGM, img_gm)
    np.save(PATHWM, img_wm)
    print('Saved file ' + PATHGM)
    print('Saved file ' + PATHWM)


## 5
for i in range(2510,3119):
    PATHGM = 'gm/gm_part5/sub'+str(i)+'_gm'
    PATHWM = 'wm/wm_part5/sub'+str(i)+'_wm'
    try:
        img_gm = nib.load(PATHGM + '.nii.gz')
        img_wm = nib.load(PATHWM + '.nii.gz')

    except OSError:
        print("File " +PATHGM+ " might be corrupted")
        continue
    except:
        print("File "+PATHGM+ " doesn't seem to exist")
        continue
    img_gm = np.array(img_gm.dataobj)
    img_wm = np.array(img_wm.dataobj)
    np.save(PATHGM, img_gm)
    np.save(PATHWM, img_wm)
    print('Saved file ' + PATHGM)
    print('Saved file ' + PATHWM)


## 6
for i in range(3120,3299):
    PATHGM = 'gm/gm_part6/sub'+str(i)+'_gm'
    PATHWM = 'wm/wm_part6/sub'+str(i)+'_wm'
    try:
        img_gm = nib.load(PATHGM + '.nii.gz')
        img_wm = nib.load(PATHWM + '.nii.gz')
    except OSError:
        print("File " +PATHGM+ " might be corrupted")
        continue
    except:
        print("File "+PATHGM+ " doesn't seem to exist")
        continue
    img_gm = np.array(img_gm.dataobj)
    img_wm = np.array(img_wm.dataobj)
    np.save(PATHGM, img_gm)
    np.save(PATHWM, img_wm)
    print('Saved file ' + PATHGM)
    print('Saved file ' + PATHWM)

