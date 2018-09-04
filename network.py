import torch
import numpy as np
import time
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict
import torchvision.models as models
import timeit
import json
import loaddata

def select_model(name='vgg19'):
    if name == 'vgg19':
        model, feature_count = models.vgg19_bn(pretrained=True), 25088
    elif name == 'densenet161':
        model, feature_count = models.densenet161(pretrained=True), 2208
    elif name == 'densenet121':
        model, feature_count = models.densenet121(pretrained=True), 1024
    else:
        model, feature_count = models.alexnet(pretrained=True), 9216

    return model, feature_count
    
def build_classifier(input_size, hidden_layer, dropout):
    orderd_dict = OrderedDict([])
    if hidden_layer == 0: 
        print("Dropout and hidden units are ignored for zero hidden layers")
    else:
        start = input_size
        layer = 1
        for i in hidden_layer:
            orderd_dict.update({'fc{}'.format(layer): nn.Linear(start, i)})
            orderd_dict.update({'relu{}'.format(layer): nn.ReLU()})
            orderd_dict.update({'do{}'.format(layer): nn.Dropout(dropout)})
            start = i
            layer += 1

    orderd_dict.update({'fcout': nn.Linear(start, 102)})
    orderd_dict.update({'output': nn.LogSoftmax(dim=1)})

    classifier = nn.Sequential(orderd_dict)
    return classifier

def validation(model, testloader, criterion, architecture):
    test_loss = 0
    accuracy = 0
    for images, labels in testloader:

        images, labels = images.to(architecture), labels.to(architecture)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def train_model(model, trainloader, testloader, criterion, optimizer, epochs=10, print_every=10, architecture='cuda'):

    model.to(architecture)

    steps = 0
    start = time.time()
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in iter(trainloader):

            steps += 1

            images, labels = images.to(architecture), labels.to(architecture)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)


            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                
                end = time.time()
                model.eval()
                
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion, architecture)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Time needed: {:.3f}.. ".format(end - start),
                    "Running Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                model.train()
                start = time.time()

def build_optimizer(model, lr):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return criterion, optimizer

def build_and_train_model(name, hidden_layer, epochs, dropout, lr, architecture, dataloader_train, dataloader_validation, dataloader_test):
    model, feature_count = select_model(name)
    model.to(architecture)
    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # replace classifier
    model.classifier = build_classifier(feature_count, hidden_layer, dropout)

    print(model)

    criterion, optimizer = build_optimizer(model, lr)

    train_model(model, dataloader_train, dataloader_validation, criterion, optimizer, epochs=epochs, architecture=architecture)

    return model




