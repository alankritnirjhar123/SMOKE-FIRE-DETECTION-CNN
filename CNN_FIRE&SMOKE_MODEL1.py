#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow


# In[ ]:


directory = r'C:\Users\Alankrit\3D Objects'


# In[ ]:


batch = 8

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1./255,
        zoom_range=(0.5,1.5),
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=(0.3,2.5),
        horizontal_flip=True,
        fill_mode='nearest')

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255) #Just normalization (no zoom etc.) for test


# In[ ]:


train_generator = train_datagen.flow_from_directory(
        os.path.join(directory,'data\\train'),
        target_size=(150, 150),
        batch_size=batch,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        os.path.join(directory,'data\\validation'),
        target_size=(150, 150),
        batch_size=batch,
        shuffle=False,
        class_mode='binary')


test_generator = test_datagen.flow_from_directory(
        os.path.join(directory,'data\\test'),
        target_size=(150, 150),
        batch_size=batch,shuffle=False,
        class_mode='binary')


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense


# In[ ]:


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last'))




model.add(Flatten())  #CONVERTING multi-D to 1D



model.add(Dense(64,kernel_regularizer='l2'))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[ ]:


from tensorflow.keras import optimizers
adam= optimizers.Adam(learning_rate=0.000001)
sgd = optimizers.SGD(lr=0.00000001,momentum=0.9)


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])


# In[ ]:


history= model.fit_generator(
        train_generator,
        steps_per_epoch=1704// batch,
        epochs=100,
        validation_data=validation_generator,
        validation_steps= 219// batch)


# In[ ]:


model.save(os.path.join(directory,'Model_CNN_1.0'))


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:


from tensorflow import keras


# In[ ]:


loaded_model = keras.models.load_model(os.path.join(directory,'Model_CNN_1.0'))


# In[ ]:


pred = loaded_model.predict_classes(test_generator)


# In[ ]:


test = test_generator.classes


# In[ ]:


f1 = f1_score(test,pred)


# In[ ]:


acc = accuracy_score(test,pred)


# In[ ]:


print(f1,acc)


# In[ ]:


#Evaluation of Model - Confusion Matrix Plot
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['UNSAFE','SAFE'],
                      title='Confusion matrix, without normalization')


# In[ ]:




