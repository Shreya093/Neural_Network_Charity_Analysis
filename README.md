# Neural_Network_Charity_Analysis

Use TensorFlow and deep learning neural networks to analyze and classify the success of charitable donations.

## Environment

Tensorflow v. 2.7.0

## Overview

1. Compare the differences between the traditional machine learning classification and regression models and the neural network models.
2. Describe the perceptron model and its components.
3. Implement neural network models using TensorFlow.
4. Explain how different neural network structures change algorithm performance.
5. Preprocess and construct datasets for neural network models.
6. Compare the differences between neural network models and deep neural networks.
7. Implement deep neural network models using TensorFlow.
8. Save trained TensorFlow models for later use.

## Purpose

A foundation, Alphabet Soup, wants to predict where to make investments. The goal is to use machine learning and neural networks to apply features on a provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup. The initial file has 34,000 organizations and a number of columns that capture metadata about each organization from past successful fundings.

## Results

## * Data Processing

1. What variable(s) are considered the target(s) for your model?

    The target for column for the model is "IS_SUCCESSFUL", which indicates whether the money was used effectively.
    
2. What variable(s) are considered to be the features for your model?

    The following variables/columns are features used as input for training and testing the model: APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL CONSIDERATIONS, ASK_AMT.
    
3. What variable(s) are neither targets nor features, and should be removed from the input data?

    The following variables are neither targets nor features and were removed: EIN, NAME.
    
## * Compiling, Training, and Evaluating the Model

1. How many neurons, layers, and activation functions did you select for your neural network model, and why?

    The neural network model has two input layers and an output layer. The model uses Relu Acitivation function as it is good at dealing with complex dataset with numerous features and efficient with time. Additional information on the model is provided below:
    
    <img width="538" alt="1" src="https://user-images.githubusercontent.com/88418201/147516980-4a90fab9-8cb4-4110-9b7e-34550113d8f0.png">
    
2. Were you able to achieve the target model performance?

    The target performance of the model was 75% but the model fell just short and achieved an accuracy score of 72.2%.
    
    <img width="644" alt="2" src="https://user-images.githubusercontent.com/88418201/147517043-311d17f0-1ca0-4acf-8b99-3b001eec5362.png">
    
3. What steps did you take to try and increase model performance?

    We took the below steps to try and increase the performance of the model :
    
    1.  We removed the noisy or unwanted features from the input dataframe by dropping the 'STATUS' and 'SPECIAL_CONSIDERATIONS' columns.
    2. We increased the number of neurons on one of the hidden layers, then we used a model with three hidden layers.
    3. We also tried a different activation function (tanh) but none of these steps helped improve the model's performance.
    
    ## Summary
    
    The deep learning neural network model did not reach the target of 75% accuracy. Considering that this target level is pretty average we could say that the model is not outperforming.
    
    Since we are in a binary classification situation, we could use a supervised machine learning model such as the Random Forest Classifier to combine a multitude of decision trees to generate a
    classified output and evaluate its performance against our deep learning model.
    
        


    
