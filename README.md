# Deep Learning Final Project

## Introduction

In this project we try to improve the performance of diabetic retinopathy detection using transfer learning by fine-tuning
models and understanding the classification results with visualizations and explainable AI. 


## Project Structure and Execution

The project consists of 5 subfolders from part_a to part_e in which the code is organized according the project tasks a to e. 
The shared folder contains reusable code extracted from the provided template code.

### part_a

### part_b

### part_c

### part_d
`part_d` contains the implementations for ensemble learning techniques and image preprocessing techniques.
Ensemble learning techniques and image preprocessing techniques are again grouped into their own folders inside `part_d`
The dataset and the model frm `part_b` are copied into `part_d` folder to avoid impacting any other parts of the project if at any time those are changed during `part_d` implementation.

`main_d.py` has the main function to execute and compare each of the ensemble techniques and image preprocessing techniques.

it supports two execution modes: (Selecting the needed `check_model_type` can be done by uncommenting the other one.)
- `check_model_type = 'individual'`
- `check_model_type = 'ensemble'`

If executed with `individual`, it will try out all image preprocessing techniques with the base models used in ensemble learning.
The results are printed on the console and at the end of the process, a line chart will visualize how the base models performed together with the image preprocessing techniques.

If executed with `ensemble`, it will try out all the image preprocessing techniques with the final ensemble learning models. The results are again displayed similarly to the `individual` execution.

`main_d_test_predictions.py` is not used to compare and validate different types of models, but only to generate final csv file to submit to Kraggle. 
Since by running `main_d.py` we already figured out `denseNet121` is the best performing model, here the idea is to try and optimize its output by using different image preprocessing pipelines. 

eg: Gaussian Blur --> Circle Crop, Circle Crop --> Gaussian Blur --> Sharpen, etc

### part_e
`part_e` contains the visualization mechanisms which are used by the other sections of the project.

### shared
`shared` folder contains code extracted from the template code mostly. It is being imported into other parts of the project and reused.

