**MODEL_1.0

Data Source:
The images obtained for train and test dataset are from a variety of sources – Kaggle & http://smoke.ustc.edu.cn/datasets.htm along with majority of test dataset belonging from Google obtained by Image Downloaded Extension in Google Chrome(allows multiple images to be downloaded from a website).


-----------------------------------
Dataset Size:


The dataset size has been experimental and thus considerably lesser than what is considered desirable while developing DEEP LEARNING MODELS.


------------------------------------
Data Augmentation:


Due to Scarcity of the Data, various Data Augmentation (on the fly) techniques were used keeping in mind which ones could be useful and which ones could make the model poor.
All images were rescaled (even in test_dataset) and resized to 150*150.
The rotation range kept within 50 degrees, brightness augmentation in the range of (0.5-2.5) and zoom in the range of (0.5-1.5) were the prominent augmentation/modification techniques along with some others.


------------------------------------
MODEL Architecture:


The model used uses Conv2d Layers with max pooling layer having - padding and strides of 2 pixels with each convolutional layer activated via RELU(Rectified Linear Unit) Function.\
After Convolution Layers the Model is Flattened so as to make it compatible to be fed into a Dense Layer with 64 nodes with a L2 Regularization applied to Kernel and a heavy dropout of 0.5 to avoid overfitiing.
Finally a sigmoid function with 1 output layer is used to map for binary classification.(Later this output will be categorized based on a 0.5 threshold).


------------------------------------
Epochs, Batch_Size & Optimizer (MODEL 1.0) :


EPOCHS- 100 (The model is first in its line, thus the model was not pushed beyond 100 epochs, even though the visualizations of training accuracy, validation accuracy, training loss and validation loss indicated the Model was yet to reach its level best).
Due to significant range of augmentation techniques used, it was made sure to keep large number of epochs as in every epoch, steps per epoch= no. of samples/batch_size to cover one range of samples but with these many augmentations each such version of a particular image would have gone through only one time in training per epoch, thus increasing no of epochs effectively increased the versions of data the model saw while training.


Batch Size was kept very low (8), as the dataset was not big

Optimizer "ADAM" with a very low learning rate – "0.000001" was used because of a very small batch size.

*Note- SGD with momentum was also experimented with but with inaccurate results and less convergence efficiency towards a solution.


------------------------------------
OBSERVATIONS:


The Model needs to be implemented with more Convolution layers (deep model)

The model could be improved further significantly by keeping some layers of this trained model and putting some additional sequential model of keras with trainable layers to capture some more significant new features.

More Data for training could have further significantly improved the performance.

More epochs would have also increased the accuracy as the loss was constantly falling at a steady rate with validation performance better than Training performance indicating no signs of overfitting in the Model on the training set.


------------------------------------
SHORTCOMINGS:


The Model would not effective in distinguishing between a person smoking(a cigarette/any other stuff one could possible imagine) VS smoke inside house due to fire.

It needs to be modified by modifying and supplying some smoking images or barbeque images or warming fireplace (or any other such situations where indoor or outdoor smoke is not hostile) in the safe directory for the model to be trained on.

-------------------------------------
