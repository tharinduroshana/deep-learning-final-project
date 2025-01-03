# Deep Learning Final Project

## Introduction

In this project we try to improve the performance of diabetic retinopathy detection using transfer learning by fine-tuning
models and understanding the classification results with visualizations and explainable AI. 

### Trained model files: 

https://drive.google.com/drive/folders/1qvQiyx9bhaUq47RTJ4lMHkZfiSHStUmU?usp=drive_link


## Project Structure and Execution

The project consists of 5 subfolders from part_a to part_e in which the code is organized according the project tasks a to e. 
The shared folder contains reusable code extracted from the provided template code.

Important: Please download and place the datasets to the root folder. The DeepDRiD dataset should be in DeepDRiD folder and APTOS dataset should be in APTOS2019 folder.

The folder structure should look like the following
```
├── APTOS2019
│   ├── test_images
│   ├── train_images
│   ├── val_images
│   ├── test.csv
│   ├── train_1.csv
│   ├── valid.csv
├── DeepDRiD
│   ├── test
│   ├── train
│   ├── val
│   ├── test.csv
│   ├── train.csv
│   ├── val.csv
├── part_a
├── part_b
├── part_c
├── part_d
├── part_e
├── shared
├── environment.yml
├── README.md
└── .gitignore
```

### Environment setup
Use the following commands to set up the conda environment and install the dependencies
```
conda env create -f environment.yml
source activate DL24
```

### part_a
`part_a` contains the implementation for fine-tuning a pretrained model using the DeepDRiD dataset.

- Navigate to part_a/ folder and Find part_a/main_a.py which is the executable file for part a.
- You need to select one from 2 available running modes. It can be either `mode = 'single'` or `mode = 'dual'`
- Next, set the `model_type`. Choose one from `resnet18`, `resnet34`, `vgg16`, `efficientnet_b0`, `densenet121`
- Run the python file either using `python main_a.py` or using the IDE.
- Output should be saved to part_a/outputs folder

### part_b
`part_b` contains the implementation for transfer learning. A model is first trained with APTOS 2019 dataset and then trained with DeepDRiD dataset.

- Navigate to part_a/ folder and Find part_b/two_staged_transfer_learning.py which is the executable file for part b.
- You need to select one from supported 5 `model_type`s. `VGG16`, `RESNET18`, `RESNET34`, `DENSENET121`, `EFFICIENTNET_B0`.
- Next, select the `training_mode`. It can be either `TrainingModes.STANDARD.value` or `training_mode = TrainingModes.PATIENT_LEVEL.value`.
- Run the python file either using `python two_staged_transfer_learning.py` or using the IDE.
- Output should be saved to part_b/.artifacts/task_b folder

### part_c
`part_c` contains the implementation for attention mechanisms. The model is trained with 3 mechanisms (self attention, channel attention, spatial attention)

- Navigate to part_c/ folder and Find part_c/attention_code.py which is the executable file for part c.
- You need to select the `attention_mode`. It could be `AttentionModes.SELF` for self attention. `AttentionModes.CHANNEL` for channel attention. `AttentionModes.SPATIAL` for spatial attention.
- Run the python file either using `python attention_code.py` or using the IDE.
- Output should be saved to part_c/.artifacts/task_c folder

### part_d
`part_d` contains the implementations for ensemble learning techniques and image preprocessing techniques.

Ensemble learning techniques (`stacking`, `boosting`, `weighted average`, `max voting`, `bagging`) and image preprocessing methods (`Ben Graham`, `Circle Cropping`, `CLAHE`, `Gaussian Blur`, `Sharpening`) are again grouped into their own folders inside `part_d`

The dataset and the model from `part_b` are copied into `part_d` folder to avoid impacting any other parts of the project if at any time those are changed during `part_d` implementation.

- Navigate to part_d/ folder and Find part_d/main_d.py which is the executable file for part d.
- You need to select the `check_model_type`. It could be `individual` or `ensemble`.
  - `individual` will try out all image preprocessing techniques with the base models (weak learners) used for ensemble learning.
  - `ensemble`, will try out all the image preprocessing techniques with the final ensemble learning models
- Run the python file either using `python main_d.py` or using the IDE.
- Output line plots should be saved to part_d/outputs folder
  - Execution results are printed on the console during the execution and at the end of the process, a line plot will visualize how the models performed together with the image preprocessing techniques.

`main_d_test_predictions.py` is not used to compare and validate different types of models, but only to generate final csv file to submit to Kaggle. 
Since by running `main_d.py` we already figured out `denseNet121` is the best performing model, here the idea is to try and optimize its output by using different image preprocessing pipelines. 

eg: Gaussian Blur --> Circle Crop, Circle Crop --> Gaussian Blur --> Sharpen, etc

### part_e
`part_e` contains the visualization mechanisms which are used by the other sections of the project. This includes `Loss and Accuracy Charts`, `Kappa Comparison for Ensemble Learning and Image Preprocessing` and `Grad-CAM` 

To execute Grad-CAM visualizations, follow the below steps.

- Navigate to part_e/ folder and Find part_e/gradcam_visualization.py which is the executable file for part e.
- You need to select one from supported 5 `model_type`s. `resnet18`, `resnet34`, `vgg16`, `efficientnet_b0`, `densenet121`.
- Then you need to select one from 3 available running modes. It can be either `mode = 'single'`, `mode = 'dual'` or `mode = 'patient_level'`
- Then attach the file path of the pretrained model to `file_path`. Please be very specific with the file path. A wrong model will not give results. It will throw errors.
- Ex: If you choose resnet18 dual mode, you have to specifically mention resnet18 dual image trained model path.
- Output should be saved to part_e/output folder

### shared
`shared` folder contains code extracted from the template code mostly. It is being imported into other parts of the project and reused.

