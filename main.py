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


import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QPushButton
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWidgets import QMessageBox
import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

"""
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 4px dashed #aaa
            }
    
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 400)
        self.setAcceptDrops(True)

        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)

        self.setLayout(mainLayout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp36/weights/last.pt',
                                   force_reload=False)
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            img = file_path
            results = model(img)
            results = results.__str__()
            split = results.split(" ")
            print(split)
            number_of_teeth = []
            string_split = ''
            plomba = re.compile('plomba')
            for item in split:
                string_split += ' '+ item
            if plomba.search(string_split): # jesli w outpucie modelu zostalo znalezione slowo plomba
                # program liczy ostatnią wartość listy (ilosc plomb) a potem ilosc zebow
                for item in split:
                    if item.isdigit():
                        number_of_teeth.append(item)
                    else:
                        pass
                number_of_teeth = [int(i) for i in number_of_teeth]
                print(str(number_of_teeth[len(number_of_teeth) - 1]))
                plomb = int(str(number_of_teeth[len(number_of_teeth) - 1]))
                self.showResults2(str(number_of_teeth[len(number_of_teeth) - 1]))
                if plomba.search(string_split):
                    number_of_teeth = [int(i) for i in number_of_teeth]

                    print(sum(number_of_teeth))
                    self.showResults(sum(number_of_teeth)- plomb)
                else:
                    pass

            else:
                for item in split:
                    if item.isdigit():
                        number_of_teeth.append(item)
                number_of_teeth = [int(i) for i in number_of_teeth]
                s = sum(number_of_teeth)
                self.showResults(s)
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))

    def showResults(self,results):
        QMessageBox.about(self,"Ilość zębów:",f"{results}")
    def showResults2(self,results):
        QMessageBox.about(self,"Ilość plomb:",f"{results}")

app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())
"""
#Kod odpowiadający za testowanie modelu
img = 'C:\\Users\\micpo\\PycharmProjects\\OBR_MED\\test_zeby\\DentalPanoramicXrays\\Images\\18.png'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp36/weights/last.pt',
                                   force_reload=False)
results = model(img)
print(results)
matplotlib.use('TkAgg')
plt.imshow(np.squeeze(results.render()))
plt.show()
"""
#linia kodu pozwalająca uruchomić trenowanie modelu w pętli dla różnych wartości parametru batch 
os.system(f"cd C:\\Users\\micpo\\PycharmProjects\\OBR_MED\\yolov5 && python train.py --img 320 --batch {i} --epochs 750 --data dataset.yml --weights yolov5s.pt --workers 2")
for i in range(1,7):
    os.system(f"cd C:\\Users\\micpo\\PycharmProjects\\OBR_MED\\yolov5 && python train.py --img 320 --batch {2**i} --epochs 750 --data dataset.yml --weights yolov5s.pt --workers 2")
"""


