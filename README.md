# Conversion to ONNX

This repository can be used to convert the native weight files (.pb, .pth, .weights) to the standard ONNX format. Currently this repository supports 3 main libraries:

- Tensorflow
- Pytorch
- Darknet

## Installation & Usage
### Dependencies:
```sh
pip install -r requirements.txt
```
### Arguments:
- **- -model_architecture:** G Model architecture. Tensorflow, Pytorch or Darknet. [REQUIRED]
- **- -input_path:** Path to  weights file. [REQUIRED]
- **- -opset_version:** Opset version which will be used for generating ONNX file. Default is **15**. 
- **- -batch_size:** Batch size of the input. Default is **1**.
- **- -dummy_input:** Expected input size to the model (For eg. channel,width,height). Default is **3, 512, 512**.
- **- -cfg_path:** Path to *.cfg* file. [REQUIRED when model_architecture is Darknet]
- **- -num_classes:** Number of classes in the input dataset. Default is **80** (COCO dataset).
### Usage:

#### For Tensorflow:
##### Step 1: Assemble the correct input data:

This repository required *saved_model* files i.e. the *.pb* file and the variables folder which contains the *.index* and the *.data* files. Sample can be found [here](https://tfhub.dev/tensorflow/efficientdet/d0/1)

##### Step 2: Run the `main.py ` file:

```sh
python main.py [ARGS]
```
**For example:**
```sh
python main.py --model_architecture tensorflow --input_path /path/to/folder/which/contains/saved_model --opset_version opset_version
```
#### For Pytorch:

##### Step 1: Edit the `model_architecture.py` file:

For conversion of pytorch models to ONNX, we need to provide the architecture of the model. To do so, import your model architecture and create an object of the same. An example of the same can be found [here](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html). By default, this repository calls the YOLOv4 model written in pytorch.

##### Step 2: Run the `main.py ` file:

```sh
python main.py [ARGS]
```
**For example:**
```sh
python main.py --model_architecture pytorch --input_path /path/to/.pth/file --opset_version opset_version --dummy_input channels,height,width --batch_size 1
```

#### For Darknet:
##### Step 1: Assemble the correct input data:

For converting darknet models into ONNX format, we will need the *.cfg* file which was used during training and the *.weights* files along with the number of classes in the input dataset.

##### Step 1: Run the `main.py ` file:


```sh
python main.py [ARGS]
```
**For example:**
```sh
python main.py --model_architecture darknet --input_path /path/to/.weights/file --cfg_path /path/to/cfg/file --num_classes num_classes --opset_version opset_version --dummy_input channels,height,width --batch_size 1
```

