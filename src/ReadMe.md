The source code in *CNN_Demo_Keras.py* can be used in the following ways:

* The complete implementation of the code trains the neural network using 1000 images and 100 images for validation, tests the trained neural network on a held-out dataset of 125 images, and performs label-specific segmentation. 

* Alternatively, a set of trained weights have also been provided. These weights have been trained on a dataset of 1000 images, and resulted in an accuracy of 94% on a held-out dest set. The user can test these weights on a randomly selected test set. This might, however, result in the scenario of testing the classifier on images used for training the classifier. The user may also test the trained network on different image datasets.

* Care should be taken to ensure the correct filepaths for loading the images and weights. 


### Usage Instructions
```

main.py <mode>

```

mode = "train", "load" 
