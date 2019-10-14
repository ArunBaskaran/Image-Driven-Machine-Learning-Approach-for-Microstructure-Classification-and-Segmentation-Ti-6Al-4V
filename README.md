# Image-Driven-Machine-Learning-Approach-for-Microstructure-Classification-and-Segmentation-Ti-6Al-4V

This is supplementary data to the manuscript submitted to the journal, 'Computational Material Science'. 

### Objective: 

The main aim of this work is to create a pipeline to efficiently extract quantifiable microstructural features using established and well-known image segmentation algorithms, such as HOG and marker-based watershed. Efficiency is obtained by passing the input images through a convolutional neural network that can classify the microstructures into one of three classes with an accuracy of >90%. Once a label is assigned to every image, they are passed to an appropriate feature segmentation function in accordance with the classified label. For example, evey image that is classified as containing a duplex microstructure is passed to a function that can extract the area fraction of globular grains using a watershed algorithm. Such a pipeline can be implemented for systems for which there is a prior knowledge of morphologies demonstrated by the system. 

The repository contains the following data:

* Raw Images: These are distributed across the folders Images1 and Images2 (because Github wouldn't allow more than a 1000 files in a sub-directory). Images has 1000 images and Images_v2 has 225 files. ## Do not treat one folder for training and the other for testing, as the images are not shuffled across classes. While running the code, make sure that the directory path for loading the images is correct. In the code provide in Pipeline_Instance.py, the images are assumed to be in the same directory as the source code. Hence, modify the line *filename = 'image_' + str(i) + '.png'*, as *filename = 'Directory/image_' + str(i) + '.png'*.

* Labels: The labels for each image in the Images1 and Images2 folders have been listed in the files labels.xlsx. Labels 1,2,and 3 refer to lamellae, bi-modal, and martensitic microstructures respectively.

* Pre-trained weights: A set of pre-trained weights for this CNN has been provided in the Weights repository. 

* Source Code for an instance of the pipeline: A sense for the pipeline can be gained from this code. It first performs the classification step and then performs label-specific segmentation tasks. Snippets of the code that have been adopted from the official OpenCV documentation have been labeled accordingly. 

