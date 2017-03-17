# Machine Learning Udacity Nanodegree
Author: @fernando.favoretti

This project was made for the nanodegree of Machine Learning in udacity

> This repository contains the following items:


    >>1.    ***CNN.ipynb***
    
    
> This is the main file that contain the CNN that I've used for this project
> This scripts generate a serie of models and log files that are necessary to made the predictions after all. These file contains the prefix 'model_cat_dog_10'.
> Also, this script generate the logs of the tensor flow for viewing in tensorboard, it will generate a folder called 'tmp/tflearn_logs/model_cat_dog_10'


    >>2.    ***Single Images Predictions.ipynb***
    
    
> This file contains the script for made single images predictions in a visual mode


    >>3.    ***Exploratory Visualization.ipynb***
    
    
> This file contains only a simple study on the part of exploratory visualization 


    >>4.    ***test_model.py***
    
    
> This file contains the same script of the Single Image Predictions but this generate the file used to submit the score for the kaggle platform


    >>5.    ***images_for_test*** dir
    
    
> This dir contain a very little sample of images that had been used to test images that the CNN was never seen before in the Single Images Predictions file


    >>6.    ***MachineLearningNanodegreeCapstoneProject.pdf*** dir
    
    
> The complete relatory for this capstone
 
 
 > The first step to run it all, is getting the kaggle dataset of the Dogs vs Cats competition:
 > They are publicly available in this link:
 
 https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
 
 
 >After, the complete path for the train and the test folder are needed to be passed to all the scripts in this repository
 
 > The first script that need to be runned is the CNN.ipynb that will generate the model files and tensorflow logs as explained above
 ***Important: The name of the model generated may be different of the used in the code, so It need to be changed in all the files in this repository that made predictions. It can be done in the 'model.load()' line***

> To successfully run all scripts these packages need to be installed:
>>1. TensorFlow: https://www.tensorflow.org/

>>2. OpenCV:
command -> pip install opencv-python
or download ->http://opencv.org/
>>3. Others libs like:
Sklearn(http://scikit-learn.org/stable/),
Scipy(https://www.scipy.org/) and
Tflearn(http://tflearn.org/installation/)
---
Thank you all!
Regards.
        

