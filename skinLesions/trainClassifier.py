import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

from PIL import Image
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

use_cuda = torch.cuda.is_available()

def train(loaders, model, criterion, optimizer, scheduler,  use_cuda, save_path, n_epochs = 12):
    """
    Train model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            outcome = model(data)
            loss = criterion(outcome,target)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        scheduler.step()
        
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            outcome = model(data)
            loss = criterion(outcome,target)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
           
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            print("Saving model...")
            torch.save(model.state_dict(), save_path)
     
    # return trained model
    return model


def test(loaders, model, criterion, use_cuda):
    '''
    Evaluate accuracy and compute confusion matrix
    '''                                          
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    y_pred = np.empty((0,1), int)
    y_true = np.empty((0,1), int)
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        y_pred = np.vstack([y_pred,pred.numpy()])
        y_true = np.vstack([y_true,target.data.view_as(pred).numpy()])
    y_pred = y_pred.squeeze()
    y_true = y_true.squeeze()
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
    print('Confusion matrix:')
    print(confusion_matrix)


def main():

    # Data loaders for training, validation, and test sets
    # Using transforms and batch_sizes

    data_dir = 'data/'
    train_dir = os.path.join(data_dir, 'train/')
    valid_dir = os.path.join(data_dir, 'valid/')
    test_dir = os.path.join(data_dir, 'test/')

    # data transformations
    data_transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224,scale=(0.75, 1.0)),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    data_transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=data_transform_train)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transform_train)
    test_data = datasets.ImageFolder(test_dir, transform=data_transform_test)

    # skin lesions classes
    classes = [strClass.replace('_',' ') for strClass in train_data.classes]

    # print out some data stats
    print('Num training images: ', len(train_data))
    print('Num validation images: ', len(valid_data))
    print('Num test images: ', len(test_data))
    print('Num of classes: ', len(classes))

    # define dataloader parameters
    batch_size = 20
    num_workers=0

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, 
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, 
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=True)

    data_loaders = {
        'train': train_loader, 
        'valid': valid_loader,
        'test': test_loader
    }

    ## Model architecture using transfer learning
    model_transfer = models.resnet18(pretrained=True)
    model_label = 'transfer_ResNet18'

    # Freeze training for all but last linear layer
    for param in model_transfer.parameters():
        param.requires_grad = False

    in_features = model_transfer.fc.in_features
    model_transfer.fc = nn.Linear(in_features, len(classes))

    if use_cuda:
        model_transfer = model_transfer.cuda()
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lr=0.001,momentum=0.9,params=model_transfer.fc.parameters())
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    model_transfer = train(
        data_loaders,
        model_transfer,
        criterion,
        optimizer,
        lr_scheduler,
        use_cuda,
        f'model_{model_label}.pt'
    )

    # load the model that got the best validation accuracy
    model_transfer.load_state_dict(torch.load(f'model_{model_label}.pt'))

    # evaluate model in test set
    test(data_loaders, model_transfer, criterion, use_cuda)

if __name__ == '__main__':
    main()