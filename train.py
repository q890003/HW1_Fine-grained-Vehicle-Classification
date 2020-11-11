import time
import config
from datasets import get_dataloader
from transforms import transform_factory
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from models import get_resnet101_32x8d_pretrained_model
import utils


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_bgin = time.time()
        print('Epoch %d/%d'%(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        time_epoch = time.time() - epoch_bgin
        print('One epoch time: {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    model_name = 'WSD_aug_ranger_'
    state = {
        'epoch': epoch + 1, 
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'sheduler': scheduler.state_dict()
    }
    return model, state


if __name__ == '__main__':
    print(config.epochs)
    dataset_loaders, dataset_loaders_size = get_dataloader()
    print(dataset_loaders_size)
    device = utils.get_gpu()
    model = get_resnet101_32x8d_pretrained_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(
        model.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-3
    )

    exp_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer_ft,
        base_lr=1e-4,
        max_lr=5e-2,
        step_size_up=5,
        mode='exp_range',
        gamma=0.85
    )

    model_ft, state = train_model(
        model,
        dataset_loaders,
        dataset_loaders_size,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=2
    )
    torch.save(
        model_ft.state_dict(),
        config.ckpt_dir + config.model_name
    )
