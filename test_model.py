import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array, load_img

model = load_model('my_model_2.h5')

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

img1 = load_img('test.png')
img2 = load_img('test2.png')
img3 = load_img('test3.png')

fig=plt.figure(1)
ax=fig.add_subplot(1,1,1)
ax.text(0.4, 0.8,'The model will now take in three pictures\n that it has never seen before.\n\nThese are examples of kids drawing shapes \n\nPlease close this box to see them',horizontalalignment='center',fontsize=14,verticalalignment='center',transform = ax.transAxes)
plt.axis('off')
plt.show()

plt.figure(2)
plt.subplot(131)
plt.imshow(img1)
plt.title('An attempt to draw a square')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.subplot(132)
plt.imshow(img2)
plt.title('An attempt to draw a triangle')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.subplot(133)
plt.title('An attempt to draw a circle')
plt.imshow(img3)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.show()


fig=plt.figure(3)
ax=fig.add_subplot(1,1,1)
ax.text(0.4, 0.8,'Now the model will \nverify whether the drawn shape matches the task\n\nPlease close this box to see them',horizontalalignment='center',fontsize=14,verticalalignment='center',transform = ax.transAxes)
plt.axis('off')
plt.show()


labels  = {"Square":0,"Triangle":1,"Circle":2}
prediction_1 = model.predict_proba(np.expand_dims(img1, axis=0))
prediction_2 = model.predict_proba(np.expand_dims(img2, axis=0))
prediction_3 = model.predict_proba(np.expand_dims(img3, axis=0))



plt.figure(4)
plt.subplot(131)
if prediction_1[0][0]:
    plt.title("Correct,\nthis is a sqaure")
else:
    plt.title("Incorrect,\nthis is NOT a sqaure")
plt.imshow(img1)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

plt.subplot(132)
if prediction_1[0][1]:
    plt.title("Correct,\nthis is a triangle")
else:
    plt.title("Incorrect,\nthis is NOT a triangle")
plt.imshow(img2)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

plt.subplot(133)
if prediction_1[0][2]:
    plt.title("Correct,\nthis is a circle")
else:
    plt.title("Incorrect,\nthis is NOT a circle")
plt.imshow(img3)
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

plt.show()
