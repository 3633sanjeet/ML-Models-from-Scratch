# Logistic Regression Implementation from Scratch

This repository contains a simple implementation of logistic regression from scratch using only basic libraries such as NumPy and Pandas. Additionally, it includes a comparison with the logistic regression implementation from the `scikit-learn` library. Feel free to explore and modify the code to fit your needs. Contributions and suggestions are welcome!

## Overview

Logistic regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome. The outcome is measured with a dichotomous variable (in which there are only two possible outcomes).

## Contents

- **Implementation of logistic regression from scratch**
  - `sigmoid` function
  - `predict` function
  - `loss` function
  - `gradient` function
  - `gradient_descent` function
- **Comparison with scikit-learn's logistic regression**
  - Custom dataset creation
  - Model training and evaluation using both scratch and sklearn implementation 

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- NumPy
- Pandas
- scikit-learn (for create a dataset on which to validate the model results and to use this library for model performance assessment)


You can install them using pip:

```bash
pip install numpy pandas scikit-learn
```

## Summary of results from Scratch vs SKlearn. 
Overall, the Accuracies, Precision, Recall values are almost same for both scratch vs sklearn implementation
Small differences in leanet weight values as we used simple implementation of gradient descent algorithm and did not use better strategies 
like dynamic learning rate etc; As the purpose was to implement logistic regression from scratch for learning purposes using only basic libraries like numpy and pandas; Objective was #### not to full optimize the scratch implementation
### *************************Scratch Logistic Results**************************************
Model intercept using Scratch Logistic implementation is: [-2.43]
Model coefficient using Scratch Logistic implementation are: [[-3.34] [ 2.12]]
Testing Accuracy of Scratch Logistic model: 90.0 %
Testing Precision of Scratch Logistic model: 84.95 %
Testing Recall of Scratch Logistic model: 92.94 %
### ***************************************************************
### *************************sklearn Logistic Results**************************************
Model intercept using sklearn Logistic implementation is: [-2.14]
Model coefficient using sklearn Logistic implementation are: [[-3.09  1.89]]
Testing Accuracy of sklearn model: 90.5 %
Testing Precision of sklearn model: 85.87 %
Testing Recall of sklearn model: 92.94 %
### ***************************************************************


## License

This project is licensed under the MIT License. You are free to use, copy, modify, and distribute this software, as long as you include the original copyright notice and this license in any substantial portions of the software.

Citation is appreciated but not required.
