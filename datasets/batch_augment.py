import numpy as np

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, ResizeTransform, SpatialTransform_2


def build_stage1_augmentation_pipeline():
    """Return the training and validation pipelines for stage 1 CBCT segmentation."""
    if Compose is None or ResizeTransform is None:
        raise ImportError('batchgenerators is required for stage1 augmentation.')

    valid_transforms = [
        ResizeTransform(target_size=(128, 128, 128), order=3, order_seg=1),
    ]

    if SpatialTransform_2 is None or MirrorTransform is None or ContrastAugmentationTransform is None or GaussianNoiseTransform is None:
        raise ImportError('batchgenerators is required for stage1 augmentation.')

    train_transforms = [
        SpatialTransform_2(
            patch_size=(128, 128, 128),
            patch_center_dist_from_border=30,
            do_elastic_deform=False,
            do_rotation=True,
            angle_x=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            angle_y=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            angle_z=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
            do_scale=True,
            scale=(0.95, 1.05),
            border_mode_data='constant',
            border_cval_data=0,
            border_mode_seg='constant',
            border_cval_seg=0,
            order_seg=1,
            order_data=3,
            random_crop=False,
            p_el_per_sample=0.0,
            p_rot_per_sample=0.5,
            p_scale_per_sample=0.5,
        ),
        MirrorTransform(axes=(0, 1, 2), p_per_sample=0.5),
        ContrastAugmentationTransform(contrast_range=(0.7, 1.3), preserve_range=True, p_per_sample=0.5),
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.5, p_per_channel=1, per_channel=False),
    ]

    return Compose(train_transforms), Compose(valid_transforms)


def build_stage2_augmentation_pipeline():
    """Return the training and validation pipeline for stage 2 landmark localization."""
    if Compose is None or ResizeTransform is None:
        raise ImportError('batchgenerators is required for stage2 augmentation.')

    valid_transforms = [
        ResizeTransform(target_size=(128, 128, 128), order=3, order_seg=1),
    ]

    if SpatialTransform_2 is None or MirrorTransform is None or ContrastAugmentationTransform is None or GaussianNoiseTransform is None:
        raise ImportError('batchgenerators is required for stage2 augmentation.')

    train_transforms = [
        # SpatialTransform_2(
        #     patch_size=(128, 128, 128),
        #     patch_center_dist_from_border=30,
        #     do_elastic_deform=False,
        #     do_rotation=True,
        #     angle_x=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
        #     angle_y=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
        #     angle_z=(-15 / 360.0 * 2 * np.pi, 15 / 360.0 * 2 * np.pi),
        #     do_scale=True,
        #     scale=(0.95, 1.05),
        #     border_mode_data='constant',
        #     border_cval_data=0,
        #     border_mode_seg='constant',
        #     border_cval_seg=0,
        #     order_seg=1,
        #     order_data=3,
        #     random_crop=False,
        #     p_el_per_sample=0.0,
        #     p_rot_per_sample=0.5,
        #     p_scale_per_sample=0.5,
        # ),
        ContrastAugmentationTransform(contrast_range=(0.7, 1.3), preserve_range=True, p_per_sample=0.5),
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.5, p_per_channel=1, per_channel=False),
    ]

    return Compose(train_transforms), Compose(valid_transforms)

