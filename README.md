## Tooth Detection GUI program

I've coded program based on algorythm YOLOv5s(github: https://github.com/ultralytics/yolov5). Program is using convolutional neural networks for
predicting number of teeth on a pantomograph. Model was trained on 42 X-ray photos of teeth. For better performance I've decided to make 7 classes
(gorny_trzonowy,dolny_trzonowy,gorny_przedtrzonowy,dolny_przedtrzonowy,gorny_przedni,dolny_przedni,plomba). 

#How to Use the Code: Image Analysis Application with YOLOv5 Model


## Prerequisites
Make sure you have the following libraries installed: 

```python
import torch
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import time
import pandas as pd
import IPython
import requests
import torchvision
import yaml
import tqdm
import seaborn
import os
import re
```
Download the pre-trained YOLOv5 model: This code utilizes a pre-trained YOLOv5 model, which you need to download. The model is available in the Ultralytics YOLOv5 repository on GitHub. Download the model.pt from

[model.pt](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/yolov5s.pt) and save it in the same folder as this code.

## Running the Application
Open a terminal or command prompt and navigate to the folder where this code is located.

Run the application using the command python file_name.py, where file_name.py is the name of the file containing this code.

Once the application is launched, a window with the label "Drop Image Here" will appear. You can drag and drop an image onto this window or use the button provided in the window to select an image for analysis.

Upon dropping the image, the application will load the YOLOv5 model from the model.pt file.

The application will then perform image analysis using the loaded model and display the analysis results in a dialog box.

If the model detects the object "plomba" (seal) in the image, two dialog boxes will be displayed: the first with the number of teeth in the image (excluding the seals), and the second with the number of detected seals.

If the model does not detect the "plomba" object in the image, a dialog box will be displayed with the number of teeth.

You can close the application window to exit the program.

### Note: Make sure the images you want to analyze are compatible with the YOLOv5 model's expectations and have the correct format (.png, .jpg).

Enjoy using the image analysis application with the YOLOv5 model!


### Results:
![zeby1.png](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/zeby1.png)
##  
![zeby2.png](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/Img/zeby2.png)
##
![zeby2.png](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/Img/zeby3.png)
##
![zeby2.png](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/Img/zeby4.png)
##
### Confusion Matrix for train set:

![conf](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/Img/confusion_matrix.png)

##
### losses, precision-recall,metrics curves:

![conf](https://github.com/micpow1cc/Pantomograph_teeth_counting/blob/main/Img/results1.png)
