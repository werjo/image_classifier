import torch
import os
from torchvision import datasets, transforms


def load_data(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    test_dir = os.path.join(base_dir, 'test')

    data_transforms_training = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # no random operations but we center crop the images and normalize the tensor
    data_transforms_testing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_transforms_validation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    image_dataset_train = datasets.ImageFolder(train_dir, transform=data_transforms_training)
    image_dataset_test = datasets.ImageFolder(test_dir, transform=data_transforms_testing)
    image_dataset_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders_train = torch.utils.data.DataLoader(image_dataset_train, batch_size=64, shuffle=True)
    dataloaders_test = torch.utils.data.DataLoader(image_dataset_test, batch_size=32, shuffle=True)
    dataloaders_validation = torch.utils.data.DataLoader(image_dataset_validation, batch_size=32, shuffle=True)

    # first
    print("Train Images:",  len(image_dataset_train.imgs), "Train Labels:", len(image_dataset_train.classes))
    print("Test Images:",  len(image_dataset_test.imgs), "Test Labels:", len(image_dataset_test.classes))
    print("Validation Images:",  len(image_dataset_validation.imgs), "Validation Labels:", len(image_dataset_validation.classes))

    # print text
    return dataloaders_train, dataloaders_test, dataloaders_validation, image_dataset_train.class_to_idx