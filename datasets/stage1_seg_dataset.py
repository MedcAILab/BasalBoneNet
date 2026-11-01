import os
import numpy as np
import SimpleITK as sitk

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from .batch_augment import build_stage1_augmentation_pipeline


def collect_stage1_pairs(image_dir, mask_dir, suffix='.nii.gz'):
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f'Missing image or mask directories: {image_dir} and {mask_dir}')

    image_names = sorted([x for x in os.listdir(image_dir) if x.endswith(suffix)])
    pairs = []
    for image_name in image_names:
        stem = image_name[:-len(suffix)] if image_name.endswith(suffix) else os.path.splitext(image_name)[0]
        mask_name = stem + suffix
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            pairs.append((os.path.join(image_dir, image_name), mask_path))

    if len(pairs) == 0:
        raise FileNotFoundError(f'No valid segmentation samples found under {image_dir} and {mask_dir}')
    return pairs


class Stage1SegDataLoader(DataLoader):
    """Stage 1 dataloader for CBCT segmentation."""

    def __init__(self, data, batch_size, target_size=(128, 128, 128), num_threads_in_multithreaded=1,
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True, infinite=True, patch_size=None):
        if patch_size is not None:
            target_size = patch_size
        super().__init__(
            data,
            batch_size,
            num_threads_in_multithreaded,
            seed_for_shuffle,
            return_incomplete,
            shuffle,
            infinite,
        )
        self.target_size = tuple(target_size)
        self.patch_size = self.target_size
        self.num_modalities = 1
        self.indices = list(range(len(data)))

    @staticmethod
    def _load_and_resize_array(path, target_size, is_label=False):
        image = sitk.ReadImage(path)
        image = sitk.DICOMOrient(image, 'RAI')

        if image.GetSize() == tuple(target_size[::-1]):
            array = sitk.GetArrayFromImage(image)
            return np.asarray(array, dtype=np.float32)

        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        output_size = [int(target_size[2]), int(target_size[1]), int(target_size[0])]
        output_spacing = [
            original_spacing[0] * original_size[0] / output_size[0],
            original_spacing[1] * original_size[1] / output_size[1],
            original_spacing[2] * original_size[2] / output_size[2],
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetSize(output_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

        resized = resampler.Execute(image)
        array = sitk.GetArrayFromImage(resized)
        return np.asarray(array, dtype=np.float32)

    @staticmethod
    def _normalize_image_by_percentiles(image, lower_percentile=0.5, upper_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lower_bound = float(np.percentile(image, lower_percentile))
        upper_bound = float(np.percentile(image, upper_percentile))

        if upper_bound <= lower_bound:
            return np.zeros_like(image, dtype=np.float32)

        clipped = np.clip(image, lower_bound, upper_bound)
        normalized = (clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
        return normalized.astype(np.float32)

    def load_patient(self, patient):
        image_path, mask_path = patient
        image = self._load_and_resize_array(image_path, self.target_size, is_label=False)
        mask = self._load_and_resize_array(mask_path, self.target_size, is_label=True)
        image = self._normalize_image_by_percentiles(image, lower_percentile=0.5, upper_percentile=99.5)

        return image, mask

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        data = np.zeros((self.batch_size, self.num_modalities, *self.target_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.target_size), dtype=np.float32)

        for i, patient in enumerate(patients_for_batch):
            image, mask = self.load_patient(patient)
            data[i, 0] = image
            seg[i, 0] = mask

        return {'data': data, 'seg': seg}


def build_stage1_generators(train_image_dir, train_label_dir, valid_image_dir, valid_label_dir, target_size=(128, 128, 128), batch_size=2, num_workers=2):

    train_pairs = collect_stage1_pairs(train_image_dir, train_label_dir)
    valid_pairs = collect_stage1_pairs(valid_image_dir, valid_label_dir)

    train_transform, valid_transform = build_stage1_augmentation_pipeline()

    train_loader = Stage1SegDataLoader(
        train_pairs,
        batch_size=batch_size,
        target_size=target_size,
        num_threads_in_multithreaded=num_workers,
        shuffle=True,
        infinite=True,
    )
    valid_loader = Stage1SegDataLoader(
        valid_pairs,
        batch_size=batch_size,
        target_size=target_size,
        num_threads_in_multithreaded=max(1, num_workers // 2),
        shuffle=False,
        infinite=True,
    )

    train_generator = MultiThreadedAugmenter(
        train_loader,
        train_transform,
        num_processes=num_workers,
        num_cached_per_queue=3,
        seeds=None,
        pin_memory=False,
    )
    valid_generator = MultiThreadedAugmenter(
        valid_loader,
        valid_transform,
        num_processes=max(1, num_workers // 2),
        num_cached_per_queue=1,
        seeds=None,
        pin_memory=False,
    )

    train_generator.restart()
    valid_generator.restart()
    return train_generator, valid_generator


