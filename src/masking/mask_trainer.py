import argparse
import os
import sys
from pathlib import Path
import torch
import torch.utils.data
import torchvision
import math
from mask_class import MaskDataset, get_model_instance_segmentation, test_model, get_transform
PIPELINE_ROOT = Path('./src').absolute()
sys.path.append(PIPELINE_ROOT.as_posix())
import utils

from engine import train_one_epoch, evaluate

ROOT = '/net/birdstore/Active_Atlas_Data/data_root/brains_info/masks'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Work on Animal')
    parser.add_argument('--animal', help='specify animal', required=False)
    parser.add_argument('--runmodel', help='run model', required=True)
    parser.add_argument('--debug', help='test model', required=False, default='false')
    parser.add_argument('--epochs', help='# of epochs', required=False, default=2)
    
    args = parser.parse_args()
    runmodel = bool({'true': True, 'false': False}[args.runmodel.lower()])
    debug = bool({'true': True, 'false': False}[args.debug.lower()])

    animal = args.animal
    epochs = int(args.epochs)
    if debug:
        test_model(ROOT, animal)
        sys.exit()

    dataset = MaskDataset(ROOT, animal, transforms = get_transform(train=True))
    dataset_test = MaskDataset(ROOT, animal, transforms = get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    test_cases = int(len(indices) * 0.15)
    dataset = torch.utils.data.Subset(dataset, indices[:-test_cases])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_cases:])
    # define training and validation data loaders
    workers = 1
    data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=2, shuffle=True, num_workers=workers,
                collate_fn=utils.collate_fn)
    data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=utils.collate_fn)
    print(f"We have: {len(indices)} examples, {len(dataset)} are training and {len(dataset_test)} testing")

    if torch.cuda.is_available(): 
        device = torch.device('cuda') 
        print('Using Nvidia graphics card GPU')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    # our dataset has two classs, tissue or 'not tissue'
    num_classes = 2
    modelpath = os.path.join(ROOT, 'mask.model.pth')
    if runmodel:
        # get the model using our helper function
        model = get_model_instance_segmentation(num_classes)
        # move model to the right device
        model.to(device)
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # 1 epoch takes 30 minutes on ratto
        for epoch in range(epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)
            torch.save(model.state_dict(), modelpath)
            print('Finished with masks')
        """
        for epoch in range(epochs):
            scaler = None
            lr_scheduler = None
            if epoch == 0:
                warmup_factor = 1.0 / 1000
                warmup_iters = min(1000, len(data_loader) - 1)

                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=warmup_factor, total_iters=warmup_iters
                )

            for images, targets in data_loader:
                counter = 0
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    loss_dict = model(images, targets)
                    print(f'loss_dict type={type(loss_dict)}, data={loss_dict}')
                    losses = sum(loss for loss in loss_dict.values())

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    optimizer.step()

                if lr_scheduler is not None:
                    lr_scheduler.step()

                print(f'Epoch {epoch} with loss: {loss_value}')

            # update the learning rate
            lr_scheduler.step()
            evaluate(model, data_loader_test, device=device)
            torch.save(model.state_dict(), modelpath)
        """


