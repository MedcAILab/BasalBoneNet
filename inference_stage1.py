import os

import numpy as np
import SimpleITK as sitk
import torch

from models.unet import UNet


def read_and_resize_volume(path, target_size):
    # Keep raw image for final output reference; use RAI-oriented image for model preprocessing.
    raw_image = sitk.ReadImage(path)
    model_image = sitk.DICOMOrient(raw_image, 'RAI')
    array = sitk.GetArrayFromImage(model_image).astype(np.float32)

    if array.shape == tuple(target_size):
        return raw_image, model_image, array

    original_size = model_image.GetSize()
    original_spacing = model_image.GetSpacing()
    output_size = [int(target_size[2]), int(target_size[1]), int(target_size[0])]
    output_spacing = [
        original_spacing[0] * original_size[0] / output_size[0],
        original_spacing[1] * original_size[1] / output_size[1],
        original_spacing[2] * original_size[2] / output_size[2],
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(model_image)
    resampler.SetOutputOrigin(model_image.GetOrigin())
    resampler.SetOutputDirection(model_image.GetDirection())
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetSize(output_size)
    resampler.SetInterpolator(sitk.sitkLinear)

    resized = resampler.Execute(model_image)
    return raw_image, model_image, sitk.GetArrayFromImage(resized).astype(np.float32)


def normalize_volume(array, lower_percentile=0.5, upper_percentile=99.5):
    array = np.asarray(array, dtype=np.float32)
    lower_bound = float(np.percentile(array, lower_percentile))
    upper_bound = float(np.percentile(array, upper_percentile))

    if upper_bound <= lower_bound:
        return np.zeros_like(array, dtype=np.float32)

    clipped = np.clip(array, lower_bound, upper_bound)
    normalized = (clipped - lower_bound) / (upper_bound - lower_bound + 1e-8)
    return normalized.astype(np.float32)


def resample_probability_to_reference(probability, reference_image, target_size):
    probability_image = sitk.GetImageFromArray(probability.astype(np.float32))
    probability_image.SetSpacing([
        reference_image.GetSpacing()[0] * reference_image.GetSize()[0] / target_size[2],
        reference_image.GetSpacing()[1] * reference_image.GetSize()[1] / target_size[1],
        reference_image.GetSpacing()[2] * reference_image.GetSize()[2] / target_size[0],
    ])
    probability_image.SetDirection(reference_image.GetDirection())
    probability_image.SetOrigin(reference_image.GetOrigin())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(probability_image)


def resample_label_to_reference(label_image, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetOutputSpacing(reference_image.GetSpacing())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetSize(reference_image.GetSize())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(label_image)


def build_stage1_model(device, base_channels=64):
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        activation='relu',
        output_activation='sigmoid',
    )
    model.to(device)
    model.eval()
    return model


def predict_one_volume(model, image_path, target_size, threshold, device):
    raw_image, model_image, resized_array = read_and_resize_volume(image_path, target_size)
    normalized_array = normalize_volume(resized_array)

    x = torch.from_numpy(normalized_array[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        probability = model(x).cpu().numpy()[0, 0]
        print(f'Predicted probability shape: {probability.shape}, min: {probability.min():.6f}, max: {probability.max():.6f}, mean: {probability.mean():.6f}')

    probability_on_model_size = sitk.GetArrayFromImage(
        resample_probability_to_reference(probability, model_image, target_size)
    ).astype(np.float32)
    mask = (probability_on_model_size >= threshold).astype(np.uint8)

    mask_image = sitk.GetImageFromArray(mask)
    mask_image.CopyInformation(model_image)

    # Convert prediction back to the raw input image space so output metadata and voxel location match the source file.
    mask_on_raw = resample_label_to_reference(mask_image, raw_image)
    return sitk.Cast(mask_on_raw, sitk.sitkUInt8)


def main(input_dir, output_dir, checkpoint, device, threshold, target_size, base_channels=64):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    device = torch.device(device)
    model = build_stage1_model(device, base_channels=base_channels)
    checkpoint_state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint_state)

    input_files = sorted([os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith(('.nii.gz', '.nii', '.mha', '.mhd'))])

    if not input_files:
        raise FileNotFoundError(f'No compatible medical image files found in {input_dir}')

    for image_path in input_files:
        output_name = os.path.basename(image_path)
        if output_name.endswith('.nii.gz'):
            output_name = output_name[:-7] + '.nii.gz'
        elif output_name.endswith('.nii'):
            output_name = output_name[:-4] + '.nii.gz'
        else:
            output_name = os.path.splitext(output_name)[0] + '.nii.gz'

        prediction = predict_one_volume(model, image_path, tuple(target_size), threshold, device)
        sitk.WriteImage(prediction, os.path.join(output_dir, output_name), True)
        print(f'Wrote prediction: {os.path.join(output_dir, output_name)}')


if __name__ == '__main__':
    target_size = (128, 128, 128)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = os.path.join('results', 'stage1', 'stage1_best.pth')

    inference_image_dir = '/CBCT_dataset/test/images'
    inference_output_dir = '/CBCT_dataset/test/inference'

    main(
        input_dir=inference_image_dir,
        output_dir=inference_output_dir,
        checkpoint=checkpoint_path,
        device=device,
        threshold=0.5,
        target_size=target_size,
        base_channels=64,
    )
