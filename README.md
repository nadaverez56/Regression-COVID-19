## Predicting with Regression - COVID-19
This here contains a regression analysis in Python.

Given COVID-19 data that contains different features such as tests by age groups, race etc. we predict the number of positive and negative tests using regression models such as Linear Regression (Least Squares), Lasso, Ridge & Polynomial Regression, as well as K-NN and Random Forest. For this we utilize Scikit-learn regression modules. 

### Data Pre-Processing

As part of pre-processing the data for the model, the following methods are executed:

•	One Hot Encoder (OHE) – In an effort of making the models more accurate, I added binary categorical features based on what day it is. This is done by applying OHE to the non-ordinal categorical feature ‘Day’. 

•	Variance Inflation Factor (VIF) – for getting an idea on the extent of multicollinearity in the features. 

•	Correlation – for checking correlation between features. 

•	Feature Selection: Backward Elimination – dropping features that can be “explained” by other features – this should lower the multicollinearity.

Correlation Matrix:

![cor_mat](https://user-images.githubusercontent.com/62807222/162811977-9280e2a9-87ba-4da9-9aaa-e1a1012a7bd2.png)

VIF shows that there is high multicollinearity in the features, and the percentage of correlated features in the data is 54%. 
By using Backward Elimination, we lower the VIFs and the correlation (the data has extremely high multicollinearity, I would say this is problematic for the Linear modeling. I tried to deal with this anyway, at least to lower it a bit). 

### Train-test split and Feature Scaling

After these steps I go on to split the data into train and test sets, 80% to the training set and 20% to the test set.
I apply Scaling to the train and test features, with the scaling centering the data around zero mean.

### Models

Models trained and considered are Linear Regression, Lasso, Ridge & Polynomial Regression, K-NN and Random Forest.

For each model we plot the error and the test data vs the prediction made by the fitted model.

![error](https://user-images.githubusercontent.com/62807222/162814473-46c4d223-275a-4080-a9d7-261826cf0afd.png)
![poly](https://user-images.githubusercontent.com/62807222/162814755-e9accaf5-67c1-4384-a3f8-434b93d19bfa.png)

**Linear Regression**: A Linear model was fitted to the training set, predicting the test set. For evaluation, I use 10-fold Cross Validation and select the mean score out of the scores as my metric. Furthermore, MSE, RMSE, R squared are calculated for evaluation.

**Polynomial Regression**: First, X_train and X_test are transformed to Polynomial Regression. For tuning the parameters (not defined hyper parameters in Pol. Reg.), I run the model in iterations by increasing degree. In each iteration, 10-fold CV is executed on the train set and returns the highest score. I chose the best degree based on the model with the best mean CV score. Best degree being 1 or 2 for different executions of the code.

I didn’t include more iterations for higher degrees basically because of run time, and it was clear that above X^2 the model performs worse.    

**Ridge & Lasso Regression**: For the penalty parameter hyper tuning (alpha/lambda), I use Repeated K-fold CV (10 splits 3 times). The GridSearch function uses the CV for finding the optimal alpha. The hyper tuning is executed in two iterations, one for each model. Each iterations returns the optimal parameter. 
After finding the best parameter for each model, a model is fit with the parameter and 10-fold CV is executed to find the best score.

**Random Forest**: For this model, hyper tuning is executed with GridSearch and Repeated 10-fold CV, just like Ridge and Lasso. Parameters tuned are the number of trees in the forest and the depth. After running the hyper tuning at first, there was a very long run time when given n_estimators range of 50,100,500,1000 and max_depth range of 1-10.

After running experiments, the grid search is given a pre-determined number of trees range (30-70 for POS and 70-120 for NEG) and max depth (ranging from 8 to 11 for POS and 8-10 for NEG), resulting in:
Positive tests RF: 70 trees and max depth of 10.
Negative tests RF: 100 trees and max depth of 9.

After finding the best parameters, a model is fit to the training data, and 10-fold CV is executed to find the best score for the Random Forest.

**K-Nearest Neighbors**: K-NN also uses Repeated 10-fold CV in a GridSearch for finding the best model parameter (K: how many neighbors will the model check). The model is fitted to the training set afterwards with the K that was found (K=1 for both models). 

### Results

Positive tests:

![pos](https://user-images.githubusercontent.com/62807222/162819039-8407e824-2027-4fbc-90b4-7e3fb773f7e6.png)


There were not very high differences between the models. However, it’s clear that Polynomial Regression has the lowest RMSE.  

Non-Positive Tests:

![neg](https://user-images.githubusercontent.com/62807222/162819240-1b292192-0c5f-45f8-907d-c887060b8e4a.png)


The models perform way better in predicting Non-positive tests. Here Polynomial Regression wins as well, and we can also see that Random Forest and K-NN’s RMSE is higher than the rest, which indicates that the other models are better.




 








