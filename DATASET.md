# Dataset Preparation and Annotation

This document describes the data contract used by the current implementation. It is separate from the main [README](README.md).


## Dataset Layout

The annotations used in the original study were created with [3D Slicer](https://www.slicer.org/). The CBCT image volumes and their corresponding annotations were then organized into the following directory structure:

```text
dataset/
├── train/
│   ├── images/
│   │   ├── case_001.nii.gz
│   │   ├── case_002.nii.gz
│   │   └── ...
│   ├── labels/
│   │   ├── case_001.nii.gz
│   │   ├── case_002.nii.gz
│   │   └── ...
│   └── landmarks/
│       ├── case_001/
│       │   ├── L1.mrk.json
│       │   ├── L2.mrk.json
│       │   ├── R1.mrk.json
│       │   └── R2.mrk.json
│       └── case_002/
│           ├── L1.mrk.json
│           ├── L2.mrk.json
│           ├── R1.mrk.json
│           └── R2.mrk.json
├── valid/
│   ├── images/
│   ├── labels/
│   └── landmarks/
└── test/
    ├── images/
    ├── labels/ 
    └── landmarks/
```

> [!NOTE]
> The image and mask for a case must have the same basename. The dataset pairing functions search for `.nii.gz` by default. SimpleITK can read other formats, but those files need matching pairing logic if they are used for training. 

### Annotation Precedure

- **Step 1:** Load the original CBCT volume and verify its orientation and spatial metadata. Preserve this metadata throughout annotation and export.

- **Step 2:** In the **Segment Editor**, create the target tooth region required by Stage 1. Export the binary segment as a NIfTI mask with the same physical extent, spacing, origin, and direction as the image.

- **Step 3:** In the **Markups** module, create one point for each of the four landmarks. Following the anatomical definitions in the paper, place each point at the center of the relevant root-bifurcation region.

- **Step 4:** Assign each landmark its correct anatomical identity. In this implementation, `1` denotes the superior (maxillary) point, `2` denotes the inferior (mandibular) point, and `L` and `R` denote the anatomical sides.

   | File | Anatomical role |
   |---|---|
   | `L1.mrk.json` | Left maxillary/superior landmark |
   | `R1.mrk.json` | Right maxillary/superior landmark |
   | `L2.mrk.json` | Left mandibular/inferior landmark |
   | `R2.mrk.json` | Right mandibular/inferior landmark |

   The loader canonicalizes the array orientation so that the `L` points appear on the right side of the array view and the `1` points are above the `2` points. This affects the array view only; it does not change the landmark names or their physical coordinates.

- **Step 5:** Save the image and mask with the same case identifier, and save each fiducial as an individual Markups JSON file in that case's landmark directory. For example:

   ```text
   images/case_001.nii.gz
   labels/case_001.nii.gz
   landmarks/case_001/L1.mrk.json
   landmarks/case_001/L2.mrk.json
   landmarks/case_001/R1.mrk.json
   landmarks/case_001/R2.mrk.json
   ```



> The code normalizes Stage 1 data to `RAI` and Stage 2 data to `LPS` internally. Do not manually flip the voxel array or rewrite the image header to compensate for this preprocessing.
> 
> Also, we provide an [example case](https://drive.google.com/drive/folders/1mwhgt4OI0AtOmkQ6yeO7uFj_sOXXsQFt?usp=sharing) to illustrate the annotation process and expected file structure.


## Training and Inference

The scripts use absolute example paths rooted at `/CBCT_dataset`. Before running them, replace these paths with the location of your dataset. The training scripts use the following directory variables:

| Script | Paths to configure |
|---|---|
| [train_stage1.py](train_stage1.py) | `train_image_dir`, `train_label_dir`, `valid_image_dir`, `valid_label_dir` |
| [train_stage2.py](train_stage2.py) | `train_image_dir`, `train_label_dir`, `train_landmark_dir`, `valid_image_dir`, `valid_label_dir`, `valid_landmark_dir` |

**Training:** Train the two stages in order. Run [train_stage1.py](train_stage1.py) to train the whole-volume segmentation model, then run [train_stage2.py](train_stage2.py) to train the landmark detector using the images, masks, and 3D Slicer annotations. The best checkpoints are saved by default as `results/stage1/stage1_best.pth` and `results/stage2/stage2_best.pth`.

**Inference:** Set the test image directory and output directory in [inference_stage1.py](inference_stage1.py), then run it with the Stage 1 checkpoint. The script writes each predicted mask as a `.nii.gz` file using the same case basename as its input image. For example, `images/case_001.nii.gz` produces `stage1_predictions/case_001.nii.gz`.

Stage 2 pairs each test image with a mask that has the exact same basename, so the Stage 1 output can be used directly without renaming. Set `inference_image_dir` in [inference_stage2.py](inference_stage2.py) to the test image directory, `inference_label_dir` to the Stage 1 output directory, and `inference_output_dir` to the desired landmark output directory. Its example paths point to the `valid` set; update them to the `test` set and your Stage 1 output directory before final testing. The script writes a case-specific folder containing `L1.mrk.json`, `L2.mrk.json`, `R1.mrk.json`, and `R2.mrk.json` for each scan.

The inference scripts create their output directories when needed. The image and mask input directories must already exist, and Stage 2 requires a non-empty mask for every test image.

## Final Checks

Before training or testing, verify that each image has a matching mask with the same basename, each training case has all four landmark files, and the landmark JSON files use LPS coordinates in millimeters. For inference, confirm that Stage 2 uses the Stage 1 prediction directory and that image and mask basenames match.
