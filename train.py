import argparse
import loaddata
import network
import torch
import os
import json

parser = argparse.ArgumentParser(description='Image classifier')

parser.add_argument('--gpu', action='store_true', default=False, help='Architecture, select if gpu is used for training')
parser.add_argument('--hidden_layers', type=int, nargs='+', default=[512], help='Number of units per hidden layer')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--model_init', default='alexnet', help='Model initialisation for feature extraction, can be vgg19, densenet121, densenet161, alexnet')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
parser.add_argument('--input_folder', default='flowers', help='Input folder for training and testing images')
parser.add_argument('--output_folder', default='out', help='Output folder where to store the model')

args = parser.parse_args()
print(args)

if args.gpu:
    # check for cuda availability
    if torch.cuda.is_available:
        print('CUDA is available, use cuda mode')
        architecture = 'cuda'
    else:
        print('Cuda is not available on this system, fallback to cpu mode')
        architecture = 'cpu'
else:
    print('Use cpu mode')
    architecture = 'cpu'

print("...")
print("Import training, test and validation set")
print("...")
dataloader_train, dataloader_test, dataloader_validation, class_to_idx_traing = loaddata.load_data('flowers')
print("...")
print("Building and traing model")
model = network.build_and_train_model(args.model_init, args.hidden_layers, args.epochs, args.dropout, args.lr, architecture, dataloader_train, dataloader_validation, dataloader_test)

print("...")
print("The model looks like:")
print(model)

# run agains test data
def test_model(model, testloader, architecture):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader_test:
            images, labels = data
            images, labels = images.to(architecture), labels.to(architecture)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# display test results
print("...")
test_model(model, dataloader_test, architecture)

# save model
print("...")
outputpath = os.path.join(args.output_folder, 'checkpoint.pth')
print("Save model checkpoint to", outputpath)
model_checkpoint = {
    'hidden_layers': args.hidden_layers,
    'model_init': args.model_init,
    'dropout': args.dropout,
    'state_dict': model.state_dict(),
    'class_to_idx': class_to_idx_traing
}

if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

torch.save(model_checkpoint, outputpath)

print("DONE")