## Pantomograph_teeth_counting

I've coded program based on algorythm YOLOv5s(github: https://github.com/ultralytics/yolov5). Program is using convolutional neural networks for
predicting number of teeth on a pantomograph. Model was trained on 42 X-ray photos of teeth. For better performance I've decided to make 7 classes
(gorny_trzonowy,dolny_trzonowy,gorny_przedtrzonowy,dolny_przedtrzonowy,gorny_przedni,dolny_przedni,plomba). 
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
