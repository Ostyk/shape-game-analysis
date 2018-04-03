import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from functions import *
import pandas as pd
import os

history = pd.read_csv('training.log')
plt.figure(1)
plt.subplot(211)
# summarize history for accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
#plt.show()
plt.subplot(212)
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()


pr = np.loadtxt('predictions.txt')
triangles_count, squares_count, circles_count = 0, 0 ,0

for root, dirs, files in os.walk('data/test/triangles'):
    triangles_count+= len(files)
for root, dirs, files in os.walk('data/test/squares'):
    squares_count+= len(files)
for root, dirs, files in os.walk('data/test/circles'):
    circles_count+= len(files)

q=int(len(pr))

threshhold = 0.9
y_pred = np.array([np.argmax(i) if np.argmax(i)>threshhold else 0 for i in pr])

y_true = np.zeros((q,3))
y_true = np.array([0]*triangles_count + [1]*squares_count + [2]*circles_count) #labels deduced from data folder

cm = confusion_matrix(y_true,y_pred)
cm_plot_labels = ['squares', 'triangles','circles']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix\n of the validation set')
