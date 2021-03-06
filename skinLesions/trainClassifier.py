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

def train(loaders, model, criterion, optimizer, scheduler,  use_cuda, save_path, n_epochs):
    """
    Train and validate model on trainset and validation datasets.
    A scheduler is used to reduce the learning rate at a fixed step of epochs.

    Args:
    loaders (dict of torch.utils.data.DataLoader data loaders)
    model (torch model)
    criterion (torch.nn criterion)
    optimizer (torch.optim optimizer)
    scheduler (torch.optim.lr_scheduler)
    use_cuda (bool)
    save_path (str)
    n_epochs (int)

    Return:
    model
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
        if scheduler is not None:
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

    # load the model that got the best validation accuracy
    model.load_state_dict(torch.load(save_path))

    # return trained model
    return model


def test(loader, model, criterion, use_cuda):
    '''
    Evaluate accuracy and compute confusion matrix

    Args:
    loader (torch.utils.data.DataLoader data loader)
    model (torch model)
    criterion (torch.nn criterion)
    use_cuda

    Return:
    None
    '''                                          
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    y_pred = np.empty((0,1), int)
    y_true = np.empty((0,1), int)
    
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
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

    # image classes
    class_names = [strClass.replace('_',' ') for strClass in train_data.classes]

    # print out some data stats
    print('Num training images: ', len(train_data))
    print('Num validation images: ', len(valid_data))
    print('Num test images: ', len(test_data))
    print('Num of classes: ', len(class_names))

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
    model_label = 'scratch_CNN'

    if model_label == 'transfer_ResNet18':
        # Use ResNet18 for feature extraction
        model = models.resnet18(pretrained=True)

        # Freeze training for all but last linear layer
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(class_names))
        model_params = model.fc.parameters()
        lr = 0.001

    elif model_label == 'transfer_VGG16':
        # Use VGG16 for feature extraction
        model = models.vgg16(pretrained=True)

        # Freeze training for all "features" layers
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(model.classifier[3].out_features,len(class_names))
        model_params = model.classifier.parameters()
        lr = 0.001

    else:
        class model_CNN(nn.Module):
            def __init__(self):
                super(model_CNN, self).__init__()
                ## Define layers of a CNN
                self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
                self.fc1 = nn.Linear(in_features=2048, out_features=512, bias=True)
                self.fc2 = nn.Linear(in_features=512, out_features=len(class_names), bias=True)
                self.dropout = nn.Dropout(0.50)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Softmax(dim=1)
                
            def forward(self, x):
                ## Define forward behavior
                x = self.conv1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv3(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.avgpool(x)
                x = x.reshape(-1,2048)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x

        # instantiate the CNN
        model = model_CNN()
        model_params = model.parameters()
        lr = 0.1
       
    if use_cuda:
        model = model.cuda()
        
    criterion = nn.CrossEntropyLoss()

    # SGD optimizer with lr scheduler
    #optimizer = optim.SGD(lr=lr,momentum=0.9,params=model_params)
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Adam optimizer without scheduler
    optimizer = optim.Adam(lr=lr,params=model_params)
    lr_scheduler = None

    # train and validate model
    model = train(
        loaders=data_loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        use_cuda=use_cuda,
        save_path=f'model_{model_label}.pt',
        n_epochs=12
    )

    # evaluate model in test set
    test(
        loader=data_loaders['test'],
        model=model,
        criterion=criterion,
        use_cuda=use_cuda
    )

if __name__ == '__main__':
    main()