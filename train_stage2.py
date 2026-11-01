import logging
import math
import os

import torch

from models.unet import UNet
from losses.focal import focal_loss
from datasets.stage2_landmark_dataset import build_stage2_generators


train_image_dir = '/CBCT_dataset/train/images'
train_label_dir = '/CBCT_dataset/train/labels'
train_landmark_dir = '/CBCT_dataset/train/landmarks'
valid_image_dir = '/CBCT_dataset/valid/images'
valid_label_dir = '/CBCT_dataset/valid/labels'
valid_landmark_dir = '/CBCT_dataset/valid/landmarks'


result_dir = os.path.join('results', 'stage2')
if not os.path.exists(result_dir):
    os.makedirs(result_dir)


def setup_logger(log_path):
    logger = logging.getLogger('stage2_training')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_num_batches(generator):
    data_loader = generator.generator
    return math.ceil(len(data_loader._data) / data_loader.batch_size)


def train_one_epoch(model, train_generator, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for _ in range(get_num_batches(train_generator)):
        batch = next(train_generator)
        image = torch.from_numpy(batch['data']).to(device, dtype=torch.float32)
        target = torch.from_numpy(batch['seg']).to(device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_size = image.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(1, total_samples)


def evaluate(model, val_generator, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for _ in range(get_num_batches(val_generator)):
            batch = next(val_generator)
            image = torch.from_numpy(batch['data']).to(device, dtype=torch.float32)
            target = torch.from_numpy(batch['seg']).to(device, dtype=torch.float32)

            output = model(image)
            loss = criterion(output, target)

            batch_size = image.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    return running_loss / max(1, total_samples)


def save_checkpoint(model, checkpoint_path):
    torch.save(model.state_dict(), checkpoint_path)


def main():
    logger = setup_logger(os.path.join(result_dir, 'train.log'))

    train_generator, valid_generator = build_stage2_generators(
        train_image_dir,
        train_label_dir,
        train_landmark_dir,
        valid_image_dir,
        valid_label_dir,
        valid_landmark_dir,
        target_size=(128, 128, 128),
        batch_size=4,
        num_workers=4,
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=4, base_channels=64, activation='relu', output_activation='sigmoid').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = focal_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    patience = 10
    best_validation_loss = float('inf')
    epochs_without_improvement = 0
    total_epochs = 1000

    for epoch in range(1, total_epochs + 1):
        train_loss = train_one_epoch(model, train_generator, optimizer, criterion, device)
        validation_loss = evaluate(model, valid_generator, criterion, device)

        scheduler.step()

        logger.info('Epoch %03d | train_loss=%.6f | validation_loss=%.6f | lr=%.6f', epoch, train_loss, validation_loss, optimizer.param_groups[0]['lr'])

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(model, os.path.join(result_dir, 'stage2_best.pth'))
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info('Early stopping at epoch %d because validation loss has not improved for %d epochs.', epoch, patience)
            break

    logger.info('Training finished.')


if __name__ == '__main__':
    main()

