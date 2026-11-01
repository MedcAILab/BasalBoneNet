import os
import json
import numpy as np
import SimpleITK as sitk

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

from .batch_augment import build_stage2_augmentation_pipeline


def gaussian_kernel_3d(radius=5, sigma=2.0):
    diameter = 2 * radius + 1
    half = (diameter - 1) / 2
    zz, yy, xx = np.ogrid[-half:half + 1, -half:half + 1, -half:half + 1]
    h = np.exp(-(xx * xx + yy * yy + zz * zz) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h.astype(np.float32)


def draw_gaussian(heatmap, center, radius=5, sigma=2.0):
    """Paste one Gaussian into ``heatmap`` with center in array order (z, y, x)."""
    z0, y0, x0 = center
    z0 = int(round(float(z0)))
    y0 = int(round(float(y0)))
    x0 = int(round(float(x0)))

    h = gaussian_kernel_3d(radius, sigma)
    z_limit, y_limit, x_limit = heatmap.shape[:3]
    z_min, z_max = min(z0, radius), min(z_limit - z0 + 1, radius + 1)
    y_min, y_max = min(y0, radius), min(y_limit - y0 + 1, radius + 1)
    x_min, x_max = min(x0, radius), min(x_limit - x0 + 1, radius + 1)

    masked_heatmap = heatmap[int(z0 - z_min):int(z0 + z_max),
                             int(y0 - y_min):int(y0 + y_max),
                             int(x0 - x_min):int(x0 + x_max)]
    masked_gaussian = h[int(radius - z_min):int(radius + z_max),
                        int(radius - y_min):int(radius + y_max),
                        int(radius - x_min):int(radius + x_max)]
    masked_gaussian = masked_gaussian[:masked_heatmap.shape[0], :masked_heatmap.shape[1], :masked_heatmap.shape[2]]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def landmarks_to_multichannel_heatmap(centers_zyx, shape=(128, 128, 128), radius=5, sigma=2.0):
    """One channel per landmark with fixed semantic order: [L1, L2, R1, R2]."""
    heatmap = np.zeros((len(centers_zyx), *shape), dtype=np.float32)
    for i, c in enumerate(centers_zyx):
        draw_gaussian(heatmap[i], c, radius, sigma)
    return heatmap


def collect_stage2_pairs(image_dir, label_dir, landmark_dir, suffix='.nii.gz'):
    if not os.path.exists(image_dir) or not os.path.exists(label_dir) or not os.path.exists(landmark_dir):
        raise FileNotFoundError(f'Missing image, label, or landmark directories: {image_dir}, {label_dir}, and {landmark_dir}')

    image_names = sorted([x for x in os.listdir(image_dir) if x.endswith(suffix)])
    pairs = []
    for image_name in image_names:
        stem = image_name[:-len(suffix)] if image_name.endswith(suffix) else os.path.splitext(image_name)[0]
        label_name = stem + suffix
        label_path = os.path.join(label_dir, label_name)
        landmark_path = os.path.join(landmark_dir, stem)

        if os.path.exists(label_path) and os.path.isdir(landmark_path):
            pairs.append((os.path.join(image_dir, image_name), label_path, landmark_path))

    if len(pairs) == 0:
        raise FileNotFoundError(f'No valid segmentation samples found under {image_dir}, {label_dir}, and {landmark_dir}')
    return pairs


class Stage2LandmarkDataloader(DataLoader):
    """Stage 2 dataloader: local ROI patch -> landmark heatmap."""

    landmark_order = ['L1', 'L2', 'R1', 'R2']

    def __init__(self, data, batch_size, target_size=(128, 128, 128), num_threads_in_multithreaded=1,
                 seed_for_shuffle=1234, return_incomplete=False, shuffle=True, infinite=True, patch_size=None,
                 crop_expand_range=(5, 20), random_expand=True, heatmap_radius=7, heatmap_sigma=2.0,
                 canonical_orientation='LPS'):
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
        self.num_landmarks = len(self.landmark_order)
        self.crop_expand_range = tuple(crop_expand_range)
        self.random_expand = random_expand
        self.heatmap_radius = int(heatmap_radius)
        self.heatmap_sigma = float(heatmap_sigma)
        self.canonical_orientation = canonical_orientation
        self.indices = list(range(len(data)))

    def _resample_image(self, image, target_size, is_label=False):
        output_size = [int(target_size[2]), int(target_size[1]), int(target_size[0])]  # xyz
        input_size = image.GetSize()  # xyz
        input_spacing = image.GetSpacing()
        output_spacing = [
            float(input_spacing[0]) * float(input_size[0]) / float(output_size[0]),
            float(input_spacing[1]) * float(input_size[1]) / float(output_size[1]),
            float(input_spacing[2]) * float(input_size[2]) / float(output_size[2]),
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(image)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetSize(output_size)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
        return resampler.Execute(image)

    def _array_to_identity_sitk(self, array_zyx):
        image = sitk.GetImageFromArray(np.asarray(array_zyx, dtype=np.float32))
        image.SetSpacing((1.0, 1.0, 1.0))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((1.0, 0.0, 0.0,
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
        return image

    def _sitk_orientation_code(self, image):
        return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(image.GetDirection())

    def _read_lps_from_mrk_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        markups = data.get('markups', [])
        if not markups:
            raise ValueError(f'No markups in {path}')

        coord = markups[0].get('coordinateSystem', 'LPS')
        if coord != 'LPS':
            raise ValueError(f'Expected LPS coordinateSystem in {path}, got {coord}')

        control_points = markups[0].get('controlPoints', [])
        if not control_points:
            raise ValueError(f'No controlPoints in {path}')

        pos = control_points[0].get('position', None)
        if pos is None or len(pos) != 3:
            raise ValueError(f'Invalid position in {path}')
        return float(pos[0]), float(pos[1]), float(pos[2])

    def _load_raw_landmark_points_lps(self, landmark_ref_path):
        """Load raw landmark points in LPS space from a directory of .mrk.json files."""
        if not os.path.isdir(landmark_ref_path):
            raise ValueError(f'landmark_path must be a directory: {landmark_ref_path}')

        raw = {}
        for name in self.landmark_order:
            json_path = os.path.join(landmark_ref_path, f'{name}.mrk.json')
            if not os.path.isfile(json_path):
                raise FileNotFoundError(f'Missing landmark file: {json_path}')
            raw[name] = self._read_lps_from_mrk_json(json_path)
        return raw

    def _enforce_fixed_landmark_orientation(self, image_zyx, mask_zyx, landmark_dict_zyx):
        """
        Keep semantic labels fixed (L1/L2/R1/R2) and flip arrays+coords if needed so that:
        - L points are on the right side of array view (x larger than R)
        - 1 points are above 2 points (z larger than 2)
        """
        required = ['L1', 'L2', 'R1', 'R2']
        for name in required:
            if name not in landmark_dict_zyx:
                raise KeyError(f'Missing landmark {name} in landmark_dict_zyx')

        d, h, w = image_zyx.shape
        out_image = image_zyx
        out_mask = mask_zyx
        out_lm = {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in landmark_dict_zyx.items()}

        # Enforce L on the right side in array coordinates (larger x).
        left_x = 0.5 * (out_lm['L1'][2] + out_lm['L2'][2])
        right_x = 0.5 * (out_lm['R1'][2] + out_lm['R2'][2])
        if left_x <= right_x:
            out_image = np.flip(out_image, axis=2).copy()
            out_mask = np.flip(out_mask, axis=2).copy()
            for name in required:
                out_lm[name][2] = float(w - 1) - out_lm[name][2]

        # Enforce "1 above 2" in array coordinates (larger z).
        one_z = 0.5 * (out_lm['L1'][0] + out_lm['R1'][0])
        two_z = 0.5 * (out_lm['L2'][0] + out_lm['R2'][0])
        if one_z <= two_z:
            out_image = np.flip(out_image, axis=0).copy()
            out_mask = np.flip(out_mask, axis=0).copy()
            for name in required:
                out_lm[name][0] = float(d - 1) - out_lm[name][0]

        return out_image, out_mask, out_lm

    def _bbox_from_mask(self, mask_zyx):
        pos = np.where(mask_zyx > 0)
        if len(pos[0]) == 0:
            return None
        zmin, zmax = int(np.min(pos[0])), int(np.max(pos[0]))
        ymin, ymax = int(np.min(pos[1])), int(np.max(pos[1]))
        xmin, xmax = int(np.min(pos[2])), int(np.max(pos[2]))
        return zmin, zmax, ymin, ymax, xmin, xmax

    def _choose_expand_margins(self):
        low, high = int(self.crop_expand_range[0]), int(self.crop_expand_range[1])
        if self.random_expand:
            return (
                np.random.randint(low, high + 1),
                np.random.randint(low, high + 1),
                np.random.randint(low, high + 1),
                np.random.randint(low, high + 1),
                np.random.randint(low, high + 1),
                np.random.randint(low, high + 1),
            )

        fixed = 10
        return fixed, fixed, fixed, fixed, fixed, fixed

    def _crop_with_mask_and_landmarks(self, image_zyx, mask_zyx, landmarks_zyx):
        bbox = self._bbox_from_mask(mask_zyx)
        if bbox is None:
            raise ValueError('Tooth mask is empty; cannot build ROI crop.')

        zmin, zmax, ymin, ymax, xmin, xmax = bbox
        dz0, dz1, dy0, dy1, dx0, dx1 = self._choose_expand_margins()

        D, H, W = image_zyx.shape
        cz0 = max(0, zmin - dz0)
        cz1 = min(D, zmax + dz1 + 1)
        cy0 = max(0, ymin - dy0)
        cy1 = min(H, ymax + dy1 + 1)
        cx0 = max(0, xmin - dx0)
        cx1 = min(W, xmax + dx1 + 1)

        cropped_image = image_zyx[cz0:cz1, cy0:cy1, cx0:cx1]
        cropped_mask = mask_zyx[cz0:cz1, cy0:cy1, cx0:cx1]

        cropped_landmarks = []
        for z, y, x in landmarks_zyx:
            zc = float(z) - float(cz0)
            yc = float(y) - float(cy0)
            xc = float(x) - float(cx0)
            cropped_landmarks.append([zc, yc, xc])

        return cropped_image, cropped_mask, np.asarray(cropped_landmarks, dtype=np.float32)

    def _resize_array_and_coords(self, image_zyx, mask_zyx, landmarks_zyx, target_size):
        in_d, in_h, in_w = image_zyx.shape
        out_d, out_h, out_w = target_size

        image_sitk = self._array_to_identity_sitk(image_zyx)
        mask_sitk = self._array_to_identity_sitk(mask_zyx)

        image_resized = self._resample_image(image_sitk, target_size, is_label=False)
        mask_resized = self._resample_image(mask_sitk, target_size, is_label=True)

        out_image = sitk.GetArrayFromImage(image_resized).astype(np.float32)
        out_mask = sitk.GetArrayFromImage(mask_resized).astype(np.float32)

        sz = float(out_d) / float(max(1, in_d))
        sy = float(out_h) / float(max(1, in_h))
        sx = float(out_w) / float(max(1, in_w))

        out_landmarks = landmarks_zyx.copy().astype(np.float32)
        out_landmarks[:, 0] *= sz
        out_landmarks[:, 1] *= sy
        out_landmarks[:, 2] *= sx

        out_landmarks[:, 0] = np.clip(out_landmarks[:, 0], 0, out_d - 1)
        out_landmarks[:, 1] = np.clip(out_landmarks[:, 1], 0, out_h - 1)
        out_landmarks[:, 2] = np.clip(out_landmarks[:, 2], 0, out_w - 1)

        return out_image, out_mask, out_landmarks

    def _normalize_image_by_percentiles(self, image, lower_percentile=0.5, upper_percentile=99.5):
        image = np.asarray(image, dtype=np.float32)
        lower_bound = float(np.percentile(image, lower_percentile))
        upper_bound = float(np.percentile(image, upper_percentile))

        if upper_bound <= lower_bound:
            return np.zeros_like(image, dtype=np.float32)

        clipped = np.clip(image, lower_bound, upper_bound)
        normalized = (clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
        return normalized.astype(np.float32)

    def load_patient(self, patient):
        image_path, mask_path, landmark_path = patient

        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)

        image = sitk.DICOMOrient(image, self.canonical_orientation)
        mask = sitk.DICOMOrient(mask, self.canonical_orientation)

        if self._sitk_orientation_code(image) != self.canonical_orientation:
            raise ValueError(f'Image orientation normalization failed: {image_path}')
        if self._sitk_orientation_code(mask) != self.canonical_orientation:
            raise ValueError(f'Mask orientation normalization failed: {mask_path}')

        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.float32)

        # Load fixed semantic landmarks (L1/L2/R1/R2) in LPS space, then convert to array coordinates.
        raw_points_lps = self._load_raw_landmark_points_lps(landmark_path)
        landmark_dict_zyx = {}
        for name in self.landmark_order:
            x, y, z = image.TransformPhysicalPointToContinuousIndex(raw_points_lps[name])
            landmark_dict_zyx[name] = [float(z), float(y), float(x)]

        # Do NOT relabel landmarks; only flip image/mask/coords together to enforce orientation consistency.
        image_array, mask_array, landmark_dict_zyx = self._enforce_fixed_landmark_orientation(image_array, mask_array, landmark_dict_zyx)

        landmarks_zyx = np.asarray([landmark_dict_zyx[name] for name in self.landmark_order], dtype=np.float32)

        roi_image, roi_mask, roi_landmarks = self._crop_with_mask_and_landmarks(image_array, mask_array, landmarks_zyx)
        out_image, out_mask, out_landmarks = self._resize_array_and_coords(roi_image, roi_mask, roi_landmarks, self.target_size)

        out_image = self._normalize_image_by_percentiles(out_image, lower_percentile=0.5, upper_percentile=99.5)
        heatmap = landmarks_to_multichannel_heatmap(
            out_landmarks,
            shape=self.target_size,
            radius=self.heatmap_radius,
            sigma=self.heatmap_sigma,
        )
        return out_image, out_mask, out_landmarks, heatmap

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        data = np.zeros((self.batch_size, self.num_modalities, *self.target_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, self.num_landmarks, *self.target_size), dtype=np.float32)

        for i, patient in enumerate(patients_for_batch):
            image, _, _, heatmap = self.load_patient(patient)
            data[i, 0] = image
            seg[i] = heatmap

        return {'data': data, 'seg': seg}


def build_stage2_generators(train_image_dir, train_label_dir, train_landmark_dir, valid_image_dir, valid_label_dir, valid_landmark_dir, target_size=(128, 128, 128), batch_size=2, num_workers=2):

    train_pairs = collect_stage2_pairs(train_image_dir, train_label_dir, train_landmark_dir)
    valid_pairs = collect_stage2_pairs(valid_image_dir, valid_label_dir, valid_landmark_dir)

    train_transform, valid_transform = build_stage2_augmentation_pipeline()

    train_loader = Stage2LandmarkDataloader(
        train_pairs,
        batch_size=batch_size,
        target_size=target_size,
        num_threads_in_multithreaded=num_workers,
        crop_expand_range=(5, 20),
        random_expand=True,
        canonical_orientation='LPS',
        shuffle=True,
        infinite=True,
    )
    valid_loader = Stage2LandmarkDataloader(
        valid_pairs,
        batch_size=batch_size,
        target_size=target_size,
        num_threads_in_multithreaded=max(1, num_workers // 2),
        crop_expand_range=(5, 20),
        random_expand=False,
        canonical_orientation='LPS',
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

