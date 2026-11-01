import json
import os

import numpy as np
import SimpleITK as sitk
import torch

from models.unet import UNet


LANDMARK_ORDER = ['L1', 'L2', 'R1', 'R2']


def collect_test_pairs(image_dir, label_dir, suffix='.nii.gz'):
	if not os.path.exists(image_dir) or not os.path.exists(label_dir):
		raise FileNotFoundError(f'Missing image or label directories: {image_dir} and {label_dir}')

	image_names = sorted([n for n in os.listdir(image_dir) if n.endswith(suffix)])
	pairs = []
	for image_name in image_names:
		stem = image_name[:-len(suffix)] if image_name.endswith(suffix) else os.path.splitext(image_name)[0]
		label_path = os.path.join(label_dir, stem + suffix)
		if os.path.exists(label_path):
			pairs.append((os.path.join(image_dir, image_name), label_path, stem))

	if not pairs:
		raise FileNotFoundError(f'No valid image-label pairs found under {image_dir} and {label_dir}')
	return pairs


def normalize_volume(array, lower_percentile=0.5, upper_percentile=99.5):
	array = np.asarray(array, dtype=np.float32)
	lower_bound = float(np.percentile(array, lower_percentile))
	upper_bound = float(np.percentile(array, upper_percentile))

	if upper_bound <= lower_bound:
		return np.zeros_like(array, dtype=np.float32)

	clipped = np.clip(array, lower_bound, upper_bound)
	normalized = (clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
	return normalized.astype(np.float32)


def build_stage2_model(device, base_channels=64):
	model = UNet(in_channels=1, out_channels=4, base_channels=base_channels, activation='relu', output_activation='sigmoid')
	model.to(device)
	model.eval()
	return model


def compute_bbox_from_mask(mask_zyx):
	pos = np.where(mask_zyx > 0)
	if len(pos[0]) == 0:
		return None
	zmin, zmax = int(np.min(pos[0])), int(np.max(pos[0]))
	ymin, ymax = int(np.min(pos[1])), int(np.max(pos[1]))
	xmin, xmax = int(np.min(pos[2])), int(np.max(pos[2]))
	return zmin, zmax, ymin, ymax, xmin, xmax


def expand_bbox(bbox, shape_zyx, margin=10):
	zmin, zmax, ymin, ymax, xmin, xmax = bbox
	d, h, w = shape_zyx
	cz0 = max(0, zmin - margin)
	cz1 = min(d, zmax + margin + 1)
	cy0 = max(0, ymin - margin)
	cy1 = min(h, ymax + margin + 1)
	cx0 = max(0, xmin - margin)
	cx1 = min(w, xmax + margin + 1)
	return cz0, cz1, cy0, cy1, cx0, cx1


def extract_roi_image(image, crop_zyx):
	cz0, cz1, cy0, cy1, cx0, cx1 = crop_zyx
	roi_size_xyz = [int(cx1 - cx0), int(cy1 - cy0), int(cz1 - cz0)]
	roi_index_xyz = [int(cx0), int(cy0), int(cz0)]
	return sitk.RegionOfInterest(image, roi_size_xyz, roi_index_xyz)


def resize_image_to_target(image, target_size_zyx, is_label=False):
	if tuple(reversed(image.GetSize())) == tuple(target_size_zyx):
		return image

	out_size_xyz = [int(target_size_zyx[2]), int(target_size_zyx[1]), int(target_size_zyx[0])]
	in_size_xyz = image.GetSize()
	in_spacing_xyz = image.GetSpacing()
	out_spacing_xyz = [
		float(in_spacing_xyz[0]) * float(in_size_xyz[0]) / float(out_size_xyz[0]),
		float(in_spacing_xyz[1]) * float(in_size_xyz[1]) / float(out_size_xyz[1]),
		float(in_spacing_xyz[2]) * float(in_size_xyz[2]) / float(out_size_xyz[2]),
	]

	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(image)
	resampler.SetOutputOrigin(image.GetOrigin())
	resampler.SetOutputDirection(image.GetDirection())
	resampler.SetOutputSpacing(out_spacing_xyz)
	resampler.SetSize(out_size_xyz)
	resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
	return resampler.Execute(image)


def resample_to_reference(image, reference, is_label=False):
	resampler = sitk.ResampleImageFilter()
	resampler.SetReferenceImage(reference)
	resampler.SetOutputOrigin(reference.GetOrigin())
	resampler.SetOutputDirection(reference.GetDirection())
	resampler.SetOutputSpacing(reference.GetSpacing())
	resampler.SetSize(reference.GetSize())
	resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
	return resampler.Execute(image)


def paste_roi_to_full(roi_image, full_reference_image, crop_zyx):
	cz0, _, cy0, _, cx0, _ = crop_zyx
	full = sitk.Image(full_reference_image.GetSize(), sitk.sitkFloat32)
	full.SetOrigin(full_reference_image.GetOrigin())
	full.SetDirection(full_reference_image.GetDirection())
	full.SetSpacing(full_reference_image.GetSpacing())

	paste_filter = sitk.PasteImageFilter()
	paste_filter.SetDestinationIndex([int(cx0), int(cy0), int(cz0)])
	paste_filter.SetSourceIndex([0, 0, 0])
	paste_filter.SetSourceSize(roi_image.GetSize())
	return paste_filter.Execute(full, sitk.Cast(roi_image, sitk.sitkFloat32))


def quadrant_argmax(heatmap, region):
	"""Search only within the ROI's 3D 4-region partition defined by x-z geometry.

	The semantic split is:
	- L1: x >= x_mid and z >= z_mid
	- L2: x >= x_mid and z < z_mid
	- R1: x < x_mid and z >= z_mid
	- R2: x < x_mid and z < z_mid

	All y voxels are retained in each region, so each landmark is searched in a 3D block.
	"""
	D, H, W = heatmap.shape
	z_mid = D // 2
	x_mid = W // 2

	z_idx = np.arange(D)[:, None, None]
	y_idx = np.arange(H)[None, :, None]
	x_idx = np.arange(W)[None, None, :]

	# Build a full 3D mask of shape (D, H, W) so broadcasting is valid.
	full_mask = np.ones((D, H, W), dtype=bool)
	if region == 'L1':
		valid = (x_idx >= x_mid) & (z_idx >= z_mid) & full_mask
	elif region == 'L2':
		valid = (x_idx >= x_mid) & (z_idx < z_mid) & full_mask
	elif region == 'R1':
		valid = (x_idx < x_mid) & (z_idx >= z_mid) & full_mask
	elif region == 'R2':
		valid = (x_idx < x_mid) & (z_idx < z_mid) & full_mask
	else:
		raise ValueError(f'Unknown region: {region}')

	arr = heatmap.copy().astype(np.float32)
	arr[~valid] = -1e9
	return np.unravel_index(np.argmax(arr), arr.shape)


def resize_index_to_original_index(peak_index, resized_size, original_size, offset_start):
	"""Map a peak index from the model-resized ROI grid back to the original crop scale.

	Parameters
	----------
	peak_index : int
		The index in the resized grid.
	resized_size : int
		The model input size along this axis, e.g. 128.
	original_size : int
		The size of the original cropped ROI along this axis.
	offset_start : int
		The start index of the crop in the original full image.
	"""
	if resized_size <= 0 or original_size <= 0:
		return float(offset_start)
	return float(offset_start) + float(peak_index) * float(original_size) / float(resized_size)


def heatmap_to_lps_point(heatmap, crop_zyx, reference_image, region, target_size=(128, 128, 128)):
	peak_zyx = quadrant_argmax(heatmap, region)

	crop_d = int(crop_zyx[1] - crop_zyx[0])
	crop_h = int(crop_zyx[3] - crop_zyx[2])
	crop_w = int(crop_zyx[5] - crop_zyx[4])
	target_d, target_h, target_w = tuple(int(v) for v in target_size)

	# 1) find the peak on the resized ROI grid
	# 2) map it back to the original crop coordinate system
	# 3) add the crop offset back to full-image coordinates
	z_full = resize_index_to_original_index(peak_zyx[0], target_d, crop_d, crop_zyx[0])
	y_full = resize_index_to_original_index(peak_zyx[1], target_h, crop_h, crop_zyx[2])
	x_full = resize_index_to_original_index(peak_zyx[2], target_w, crop_w, crop_zyx[4])

	index_xyz = (int(round(x_full)), int(round(y_full)), int(round(z_full)))
	physical_xyz = reference_image.TransformIndexToPhysicalPoint(index_xyz)
	return float(physical_xyz[0]), float(physical_xyz[1]), float(physical_xyz[2])


def make_slicer_markup_json(name, lps_position):
	return {
		"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
		"markups": [
			{
				"type": "Fiducial",
				"coordinateSystem": "LPS",
				"coordinateUnits": "mm",
				"locked": False,
				"fixedNumberOfControlPoints": False,
				"labelFormat": "%N-%d",
				"lastUsedControlPointNumber": 1,
				"controlPoints": [
					{
						"id": "1",
						"label": name,
						"description": "",
						"position": [float(lps_position[0]), float(lps_position[1]), float(lps_position[2])],
						"orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
						"selected": True,
						"locked": False,
						"visibility": True,
						"positionStatus": "defined"
					}
				],
				"measurements": [],
				"display": {
					"visibility": True,
					"opacity": 1.0,
					"color": [0.4, 1.0, 1.0],
					"selectedColor": [0.9019607843137255, 0.8627450980392157, 0.27450980392156865],
					"activeColor": [0.4, 1.0, 0.0],
					"propertiesLabelVisibility": False,
					"pointLabelsVisibility": True,
					"textScale": 3.0,
					"glyphType": "Sphere3D",
					"glyphScale": 3.0,
					"glyphSize": 5.0,
					"useGlyphScale": True,
					"sliceProjection": False,
					"sliceProjectionUseFiducialColor": True,
					"sliceProjectionOutlinedBehindSlicePlane": False,
					"sliceProjectionColor": [1.0, 1.0, 1.0],
					"sliceProjectionOpacity": 0.6,
					"lineThickness": 0.2,
					"lineColorFadingStart": 1.0,
					"lineColorFadingEnd": 10.0,
					"lineColorFadingSaturation": 1.0,
					"lineColorFadingHueOffset": 0.0,
					"handlesInteractive": False,
					"translationHandleVisibility": True,
					"rotationHandleVisibility": True,
					"scaleHandleVisibility": True,
					"interactionHandleScale": 3.0,
					"snapMode": "toVisibleSurface"
				}
			}
		]
	}


def predict_one_case(model, image_path, label_path, device, target_size=(128, 128, 128), crop_margin=10):
	raw_image = sitk.ReadImage(image_path)
	raw_label = sitk.ReadImage(label_path)

	model_image = sitk.DICOMOrient(raw_image, 'LPS')
	model_label = sitk.DICOMOrient(raw_label, 'LPS')

	model_image_arr = sitk.GetArrayFromImage(model_image).astype(np.float32)
	model_label_arr = sitk.GetArrayFromImage(model_label).astype(np.float32)

	bbox = compute_bbox_from_mask(model_label_arr)
	if bbox is None:
		raise ValueError(f'Empty label mask: {label_path}')

	crop_zyx = expand_bbox(bbox, model_image_arr.shape, margin=crop_margin)
	roi_image = extract_roi_image(model_image, crop_zyx)

	roi_resized = resize_image_to_target(roi_image, target_size, is_label=False)
	roi_resized_arr = sitk.GetArrayFromImage(roi_resized).astype(np.float32)
	roi_resized_arr = normalize_volume(roi_resized_arr)

	x = torch.from_numpy(roi_resized_arr[None, None].astype(np.float32)).to(device)
	with torch.no_grad():
		pred = model(x).cpu().numpy()[0]  # [4, D, H, W]

	outputs = {}
	for ch, name in enumerate(LANDMARK_ORDER):
		hm_target = pred[ch].astype(np.float32)
		peak_lps = heatmap_to_lps_point(hm_target, crop_zyx, model_image, region=name, target_size=target_size)
		outputs[name] = peak_lps

	return outputs


def main(input_image_dir, input_label_dir, output_dir, checkpoint, device='cuda:0', target_size=(128, 128, 128), crop_margin=10, base_channels=64):
	os.makedirs(output_dir, exist_ok=True)

	if not os.path.exists(checkpoint):
		raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

	pairs = collect_test_pairs(input_image_dir, input_label_dir)

	device = torch.device(device)
	model = build_stage2_model(device, base_channels=base_channels)
	checkpoint_state = torch.load(checkpoint, map_location=device)
	model.load_state_dict(checkpoint_state)

	for image_path, label_path, case_id in pairs:
		predicted_points = predict_one_case(
			model,
			image_path,
			label_path,
			device,
			target_size=tuple(target_size),
			crop_margin=int(crop_margin),
		)

		case_output_dir = os.path.join(output_dir, case_id)
		os.makedirs(case_output_dir, exist_ok=True)
		for name, point in predicted_points.items():
			json_path = os.path.join(case_output_dir, f'{name}.mrk.json')
			with open(json_path, 'w', encoding='utf-8') as f:
				json.dump(make_slicer_markup_json(name, point), f, ensure_ascii=False, indent=2)
		print(f'Wrote 4 Slicer-style mrk.json files for: {case_id}')


if __name__ == '__main__':
	inference_image_dir = '/CBCT_dataset/valid/images'
	inference_label_dir = '/CBCT_dataset/valid/labels'    # replace with stage1 tooth mask output path for testing
	inference_output_dir = '/CBCT_dataset/valid/inference'
	checkpoint_path = os.path.join('results', 'stage2', 'stage2_best.pth')

	main(
		input_image_dir=inference_image_dir,
		input_label_dir=inference_label_dir,
		output_dir=inference_output_dir,
		checkpoint=checkpoint_path,
		device='cuda:0' if torch.cuda.is_available() else 'cpu',
		target_size=(128, 128, 128),
		crop_margin=10,
		base_channels=64,
	)
