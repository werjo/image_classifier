import torch
import argparse
import network
import json
import loaddata
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Image predicter')

parser.add_argument('input', help='path to checkpoint')
parser.add_argument('path_to_image', help='path to image to predict')
parser.add_argument('--gpu', action="store_true", default=False, help='Utilize gpu for predictions, default is false')
parser.add_argument('--top_k', type=int, default=5, help='print top k classe, default is 5')
parser.add_argument('--category_names', default="cat_to_name.json", help='A json file to map class names to the output')


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


#load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # print(checkpoint)

    model, features = network.select_model(checkpoint['model_init'])
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = network.build_classifier(features, checkpoint['hidden_layers'], checkpoint['dropout'])
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

model = load_checkpoint(args.input)



# transforms to test the image
tranform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image(path_to_image):
    image = Image.open(path_to_image)
    img_tensor = tranform(image)
    img_variable = img_tensor.unsqueeze_(0)
    img_variable = img_tensor.float()
    return img_variable

def predict(image_path, model, topk, architecture):   
    
    img = load_image(image_path)

    model.eval()   
    # set architecture (cuda or cpu)
    model.to(architecture)
    img = img.to(architecture)

    with torch.no_grad():
        output = model.forward(img)
        
    # get props
    probability = torch.exp(output.data)
    
    # get top k procs
    top_probs, top_labs = probability.topk(topk)

    # convert to numpy lists
    top_probs = top_probs.cpu().numpy()[0].tolist()
    top_labs = top_labs.cpu().numpy()[0].tolist()

    # reverse class_to_idx
    idx_to_class = {val: key for key, val in model.class_to_idx.items() }

    # map to classes from file and to string labels
    top_labels = [idx_to_class[label] for label in top_labs]
    top_flowers = [cat_to_name[idx_to_class[label]] for label in top_labs]

    return top_probs, top_labels, top_flowers


probs, labels, flowers = predict(args.path_to_image, model, args.top_k, architecture)

print("...")
print("Your picture is classified as '{}'".format(flowers[0]))
print("Here are the top {} classes:".format(args.top_k))
for i in range(len(probs)):
    print("{}. Image is classified as".format(i), "'{}'".format(flowers[i]), "with probability:", "{:.1f}%".format(probs[i] * 100))

# map categories
print("DONE")