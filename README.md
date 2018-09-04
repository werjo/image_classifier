# Data Scientist Project

Project code for Udacity's Data Scientist Nanodegree program. In this project, you will first develop code for an image classifier built with PyTorch, then you will convert it into a command line application.

In order to complete this project, you will need to use the GPU enabled workspaces within the classroom.  The files are all available here for your convenience, but running on your local CPU will likely not work well.

You should also only enable the GPU when you need it. If you are not using the GPU, please disable it so you do not run out of time!

### Data

The data for this project is quite large - in fact, it is so large you cannot upload it onto Github.  If you would like the data for this project, you will want download it from the workspace in the classroom.  Though actually completing the project is likely not possible on your local unless you have a GPU.  You will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.

### Project Setup

python Version: 3.6.5
Conda Version: 4.5.11

### Run train.py
```
optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Architecture, select if gpu is used for training
  --hidden_layers HIDDEN_LAYERS [HIDDEN_LAYERS ...]
                        Number of units per hidden layer
  --lr LR               Learning rate
  --model_init MODEL_INIT
                        Model initialisation for feature extraction, can be
                        vgg19, densenet121, densenet161, alexnet
  --epochs EPOCHS       Number of epochs
  --dropout DROPOUT     Dropout
  --input_folder INPUT_FOLDER
                        Input folder for training and testing images
  --output_folder OUTPUT_FOLDER
                        Output folder where to store the model

```

Example: python train.py --gpu --model_init vgg19 --hidden_layers 4000 2000 500 --dropout=0.33 --epochs=8 --lr=0.001

### Run predict.py
```
positional arguments:
  input                 path to checkpoint
  path_to_image         path to image to predict
optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Utilize gpu for predictions, default is false
  --top_k TOP_K         print top k classe, default is 5
  --category_names CATEGORY_NAMES
                        A json file to map class names to the output

```

Example: python predict.py out/checkpoint.pth flowers/test/46/image_00958.jpg --gpu --top_k=3

