# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
import warnings
warnings.filterwarnings('ignore')

# Functions that will be used:
    
def correlation_check(X):
    correlated_features = set()
    corr_matrix = X.corr()
    #print(corr_matrix)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.9: #threshhold chosen: 0.9
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    return len(correlated_features)/len(X.columns), corr_matrix
                 
def corr_mat(X):
    f = plt.figure(figsize=(19, 15))
    plt.matshow(X.corr(),fignum=f.number)
    plt.xticks(range(X.select_dtypes(['number']).shape[1]), X.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(X.select_dtypes(['number']).shape[1]), X.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()

#feature-selection - B.E
def backwardElimination(X, y, Sig_level):
    X_arr = X.values
    names = X.columns.tolist()    
    num_of_vars = len(X_arr[0])
    for i in range(0, num_of_vars):
        regressor_OLS = sm.OLS(y, X_arr).fit()
        highest_SL = max(regressor_OLS.pvalues).astype(float)
        if highest_SL > Sig_level:
            for j in range(0, num_of_vars - i):
                if (regressor_OLS.pvalues[j].astype(float) == highest_SL):
                    X_arr = np.delete(X_arr, j, 1)
                    del names[j]
                    break
    
    regressor_OLS.summary()
    return X_arr, names

def VIF_check(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
# calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
    return vif_data

def model_eval(y_test,y_pred, CV_score,polynomial=False):    
   
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = 'No R2 - it is Poly' if polynomial else r2_score(y_test, y_pred)
    model_metrices = [mse, rmse, r2, CV_score]    
    print("The model performance for testing set")
    print("--------------------------------------")
    print('MSE: {}'.format(mse))
    print('RMSE: {}'.format(rmse))
    print('R2 score: {}'.format(r2))    
    return model_metrices

def actualVSpredicted(y_pred, y_test, model_type):
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, edgecolors=(0, 0, 1))
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
    fig.suptitle('Actual vs Predicted - '+ model_type, fontsize=15)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.show()

def histogram_(error):
    plt.hist(error)
    plt.suptitle("Error (Test - Predicted)")
    
# ============================================================================
PATH = 'COVID-19_Daily_Testing_-_By_Test.csv'
data = pd.read_csv(PATH)
data = data.dropna().drop_duplicates() # only 1 NA value - we'll drop the row

#columns F-W + day column (adding this to the model)
x = data.iloc[:, 5:22]
x['Day'] = data['Day']
X = x.values

column_names = x.columns.values
enc = OneHotEncoder(sparse=False)
day_onehot = enc.fit_transform(pd.DataFrame(X).loc[:,[17]])
#to print the encoded features for train data
OHE_df = pd.DataFrame(day_onehot, columns=list(enc.categories_[0]))

weekdays_col = pd.DataFrame(day_onehot, columns=list(enc.categories_[0])).columns.values

X = pd.concat([pd.DataFrame(X).drop(17,1), pd.DataFrame(day_onehot)],axis=1)

#adding column names back
feature_names = np.concatenate([column_names,weekdays_col])
feature_names = np.delete(feature_names, np.where(feature_names == 'Day'))
X.columns = feature_names

# all int features were casted to object type
X = X.astype(int) 


y_pos = data.iloc[:, 2].values # 2 - first response var.
y_neg = data.iloc[:, 3].values # 3 - second response var.
#correlation check
    
# VIF for showing multicollinearity of the features:
VIF_check(X)

# most of the features have >5 VIF (INF!). We can see that the binary weekdays features have OK VIF,
# which indicates that it was a smart move to use OHE.
# for trying to tackle this issue, we'll try correlation check:
# first we'll see the correlation matrix before feature selection:

print("Perentage of correlated features out of data: ", correlation_check(X)[0], "%")

# now after implementing Backward Elimination:
X_arr,names = backwardElimination(X, y_pos, 0.05)  
X_arr1, names1 = backwardElimination(X, y_neg, 0.05)

X_pos = pd.DataFrame(X_arr)
X_neg = pd.DataFrame(X_arr1)

X_pos.columns = names
X_neg.columns = names1

# corr_mat(X)
# print(correlation_check(X)[1])
print("Perentage of correlated features out of data: ", correlation_check(X_pos)[0], "%")
print("Perentage of correlated features out of data: ", correlation_check(X_neg)[0], "%")


# VIF after feature selection:
    
VIF_check(X_pos)
VIF_check(X_neg)


# we can see that it helped a bit in lowering the VIF, but not perfect.

# =============================================================================

## splitting to train/test

X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(X_pos, y_pos, test_size = 0.2, random_state = 0)
X_neg_train, X_neg_test, y_neg_train, y_neg_test = train_test_split(X_neg, y_neg, test_size = 0.2, random_state = 0)

scaler_pos = StandardScaler()
scaler_neg = StandardScaler()

X_pos_train = scaler_pos.fit_transform(X_pos_train)
X_neg_train = scaler_neg.fit_transform(X_neg_train)

X_pos_test = scaler_pos.transform(X_pos_test)
X_neg_test = scaler_neg.transform(X_neg_test)
# then training the model

# ============================================================================
LR_Constructors = [LinearRegression(),LinearRegression()]
# Linear Regression
# Fitting Multivariate Linear Regression to the training set
regressor_pos,regressor_neg = LR_Constructors

regressor_pos.fit(X_pos_train, y_pos_train)
y_pred_pos = regressor_pos.predict(X_pos_test)

regressor_neg.fit(X_neg_train, y_neg_train)
y_pred_neg = regressor_neg.predict(X_neg_test)


error_pos = y_pos_test-y_pred_pos
error_neg = y_neg_test-y_pred_neg

b_0_pos=regressor_pos.intercept_
b_1_pos=regressor_pos.coef_

b_0_neg=regressor_neg.intercept_
b_1_neg=regressor_neg.coef_

# model evaluation:
score_pos = cross_val_score(regressor_pos, X_pos_train, y_pos_train, cv=10)
score_neg = cross_val_score(regressor_neg, X_neg_train, y_neg_train, cv=10)

LinearReg_metrices_pos = model_eval(y_pos_test,y_pred_pos,np.mean(score_pos),False) 
LinearReg_metrices_neg = model_eval(y_neg_test,y_pred_neg,np.mean(score_neg),False) 

histogram_(error_pos)
histogram_(error_neg)

print('Intercept POS: {}'.format(b_0_pos))
print('Coefficients POS: {}'.format(b_1_pos))
print('Intercept NEG: {}'.format(b_0_neg))
print('Coefficients NEG: {}'.format(b_1_neg))


actualVSpredicted(y_pred_pos, y_pos_test,'Linear Regression')
actualVSpredicted(y_pred_neg, y_neg_test,'Linear Regression')

#=============================================================================

# Polynomial Regression

degrees = list(range(1,4)) # Change degree "hyperparameter" here
best_score_pos = 0
best_score_neg = 0
best_degree_pos = 0
best_degree_neg = 0

for degree in degrees:    
        poly = PolynomialFeatures(degree = degree)
        X_pos_train_p = poly.fit_transform(X_pos_train)
        X_pos_test_p = poly.fit_transform(X_pos_test)
        polynomial_regressor_pos = LR_Constructors[0]
        polynomial_regressor_pos.fit(X_pos_train_p, y_pos_train)
        y_pred_pos = polynomial_regressor_pos.predict(X_pos_test_p)
        error_pos = (y_pred_pos-y_pos_test)
        scores_pos = cross_val_score(polynomial_regressor_pos, X_pos_train_p, y_pos_train, cv=10) # Change k-fold cv value here    
        if np.mean(scores_pos) > best_score_pos:
            best_score_pos = np.mean(scores_pos)
            best_degree_pos = degree
            best_y_pred_pos = y_pred_pos
print(best_score_pos)
print(best_degree_pos)
histogram_(error_pos)
Poly_metrices_pos = model_eval(y_pos_test,best_y_pred_pos,best_score_pos,True) 
actualVSpredicted(best_y_pred_pos, y_pos_test,'Polynomial Regression')

for degree in degrees:    
        poly = PolynomialFeatures(degree = degree)
        X_neg_train_p = poly.fit_transform(X_neg_train)
        X_neg_test_p = poly.fit_transform(X_neg_test)
        X_neg_train_p = poly.fit_transform(X_neg_train)
        X_neg_test_p = poly.fit_transform(X_neg_test)
        polynomial_regressor_neg = LR_Constructors[0]
        polynomial_regressor_neg.fit(X_neg_train_p, y_neg_train)
        y_pred_neg = polynomial_regressor_neg.predict(X_neg_test_p)
        error_neg = (y_pred_neg-y_neg_test)
         # Change k-fold cv value here
        scores_neg = cross_val_score(polynomial_regressor_neg, X_neg_train_p, y_neg_train, cv=10)
        if np.mean(scores_neg) > best_score_neg:
            best_score_neg = np.mean(scores_neg)
            best_degree_neg = degree
            best_y_pred_neg = y_pred_neg
print(best_score_neg)
print(best_degree_neg)
histogram_(error_neg)
Poly_metrices_neg = model_eval(y_neg_test,best_y_pred_neg,best_score_neg,True) 
actualVSpredicted(best_y_pred_neg, y_neg_test,'Polynomial Regression')


#=============================================================================

#Ridge & Lasso Regression Hyperparameter Tuning - we will use Grid Search 
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
grid = dict()
grid['alpha'] = np.arange(0,1,0.01)
models = [Ridge,Lasso]
alphas_pos = []
alphas_neg = []
#NEG
for i in range(0,2):  
    model_regressor = models[i]()
    search = GridSearchCV(model_regressor, grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    results = search.fit(X_pos_train, y_pos_train)
    alphas_pos.append(results.best_params_['alpha'])
#NEG
for i in range(0,2):  
    model_regressor = models[i]()
    search = GridSearchCV(model_regressor, grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    results = search.fit(X_neg_train, y_neg_train)
    alphas_neg.append(results.best_params_['alpha'])
    
# training the models by our optimal parameters
#POS
Ridge_regressor_pos = Ridge(alpha=alphas_pos[0])
Ridge_regressor_pos.fit(X_pos_train, y_pos_train)
y_pred_pos = Ridge_regressor_pos.predict(X_pos_test)
error_pos = y_pred_pos-y_pos_test
histogram_(error_pos)
scores_ridge_pos = cross_val_score(Ridge_regressor_pos, X_pos_train, y_pos_train, cv=10) # scoring = R2, by default
Ridge_metrices_pos = model_eval(y_pos_test,y_pred_pos,np.mean(scores_ridge_pos),False) 
actualVSpredicted(y_pred_pos, y_pos_test,'Ridge Regression')
#NEG
Ridge_regressor_neg = Ridge(alpha=alphas_neg[0])
Ridge_regressor_neg.fit(X_neg_train, y_neg_train)
y_pred_neg = Ridge_regressor_neg.predict(X_neg_test)
error_neg = y_pred_neg-y_neg_test
histogram_(error_neg)
scores_ridge_neg = cross_val_score(Ridge_regressor_neg, X_neg_train, y_neg_train, cv=10) # scoring = R2, by default
Ridge_metrices_neg = model_eval(y_neg_test,y_pred_neg,np.mean(scores_ridge_neg),False) 
actualVSpredicted(y_pred_neg, y_neg_test,'Ridge Regression')

#POS
Lasso_regressor_pos = Lasso(alpha=alphas_pos[1])
Lasso_regressor_pos.fit(X_pos_train, y_pos_train)
y_pred_pos = Lasso_regressor_pos.predict(X_pos_test)
error_pos = y_pred_pos-y_pos_test
histogram_(error_pos)
scores_lasso_pos = cross_val_score(Lasso_regressor_pos, X_pos_train, y_pos_train, cv=10) # scoring = R2, by default
Lasso_metrices_pos = model_eval(y_pos_test,y_pred_pos,np.mean(scores_lasso_pos),False) 
actualVSpredicted(y_pred_pos, y_pos_test,'Lasso Regression')
#NEG
Lasso_regressor_neg = Lasso(alpha=alphas_neg[1])
Lasso_regressor_neg.fit(X_neg_train, y_neg_train)
y_pred_neg = Lasso_regressor_neg.predict(X_neg_test)
error_neg = y_pred_neg-y_neg_test
histogram_(error_neg)
scores_lasso_neg = cross_val_score(Lasso_regressor_neg, X_neg_train, y_neg_train, cv=10) # scoring = R2, by default
Lasso_metrices_neg = model_eval(y_neg_test,y_pred_neg,np.mean(scores_lasso_neg),False) 
actualVSpredicted(y_pred_neg, y_neg_test,'Lasso Regression')

#=============================================================================

# Random Forest
# after running the hyper tuning check with a very long run time, the grid search for hyper tuning will be given number of trees in the 30-70 range, 
# and depth between 8 to 11.

rf_pos,rf_neg = [RandomForestRegressor(random_state=0),RandomForestRegressor(random_state=0)]

#POS
tuning_parameters_pos = {'n_estimators': [30, 50, 70], 'max_depth': [8,9,10,11]}

search_pos = GridSearchCV(rf_pos, tuning_parameters_pos, cv=cv, n_jobs=-1)
results_pos = search_pos.fit(X_pos_train,y_pos_train)

rf_pos = RandomForestRegressor(n_estimators=results_pos.best_params_['n_estimators'],
                           max_depth=results_pos.best_params_['max_depth'],random_state=0)
rf_pos.fit(X_pos_train, y_pos_train)
y_pred_pos = rf_pos.predict(X_pos_test)
error_pos = y_pred_pos-y_pos_test
histogram_(error_pos)
scores_rf_pos = cross_val_score(rf_pos, X_pos_train, y_pos_train, cv=10) # scoring = R2, by default
RF_metrices_pos = model_eval(y_pos_test,y_pred_pos,np.mean(scores_rf_pos),False)
actualVSpredicted(y_pred_pos, y_pos_test,'Random Forest')

# NEG
tuning_parameters_neg = {'n_estimators': [70, 100, 120], 'max_depth': [8,9,10]}

search_neg = GridSearchCV(rf_neg, tuning_parameters_neg, cv=cv, n_jobs=-1)
results_neg = search_neg.fit(X_neg_train,y_neg_train)
results_neg.best_params_

rf_neg = RandomForestRegressor(n_estimators=results_neg.best_params_['n_estimators'],
                           max_depth=results_neg.best_params_['max_depth'],random_state=0)
rf_neg.fit(X_neg_train, y_neg_train)
y_pred_neg = rf_neg.predict(X_neg_test)
error_neg = y_pred_neg-y_neg_test
histogram_(error_neg)
scores_rf_neg = cross_val_score(rf_neg, X_neg_train, y_neg_train, cv=10) # scoring = R2, by default
RF_metrices_neg = model_eval(y_neg_test,y_pred_neg,np.mean(scores_rf_neg),False)
actualVSpredicted(y_pred_neg, y_neg_test,'Random Forest')


#=============================================================================

# K-Nearest Neighbors 
#POS
knn_pos, knn_neg = [KNeighborsClassifier(),KNeighborsClassifier()]
grid = dict(n_neighbors=(range(1, 10)))

search_pos = GridSearchCV(knn_pos, grid, cv=cv, scoring="neg_mean_squared_error")
results_pos = search_pos.fit(X_pos_train, y_pos_train) 

knn_pos = KNeighborsClassifier(n_neighbors = results_pos.best_params_['n_neighbors']) #k=1
knn_pos.fit(X_pos_train, y_pos_train)
y_pred_pos = knn_pos.predict(X_pos_test)
error_pos = y_pred_pos-y_pos_test
histogram_(error_pos)
knn_metrices_pos = model_eval(y_pos_test,y_pred_pos,'No CV Score',False)
actualVSpredicted(y_pred_pos, y_pos_test,'K-Nearest Neighbors')

#NEG
grid = dict(n_neighbors=(range(1, 10)))

search_neg = GridSearchCV(knn_neg, grid, cv=cv, scoring="neg_mean_squared_error")
results_neg = search_neg.fit(X_neg_train, y_neg_train) 

knn_neg = KNeighborsClassifier(n_neighbors = results_neg.best_params_['n_neighbors']) #k=1
knn_neg.fit(X_neg_train, y_neg_train)
y_pred_neg = knn_neg.predict(X_neg_test)
error_neg = y_pred_neg-y_neg_test
histogram_(error_neg)
knn_metrices_neg = model_eval(y_neg_test,y_pred_neg,'No CV Score',False)
actualVSpredicted(y_pred_neg, y_neg_test,'K-Nearest Neighbors')


first_Models_results = pd.DataFrame({'Linear Regression': LinearReg_metrices_pos, 'Polynomial Regression': Poly_metrices_pos,
                   'Ridge Regression': Ridge_metrices_pos, 'Lasso Regression': Lasso_metrices_pos,
                   'Random Forest': RF_metrices_pos,'K-Nearest Neighbors': knn_metrices_pos})
metrices = ['MSE','RMSE','R^2','CV_score']
first_Models_results.insert(loc=0,column='Metric',value=metrices)
print(first_Models_results)

second_Models_results = pd.DataFrame({'Linear Regression': LinearReg_metrices_neg, 'Polynomial Regression': Poly_metrices_neg,
                   'Ridge Regression': Ridge_metrices_neg, 'Lasso Regression': Lasso_metrices_neg,
                   'Random Forest': RF_metrices_neg,'K-Nearest Neighbors': knn_metrices_neg})
metrices = ['MSE','RMSE','R^2','CV_score']
second_Models_results.insert(loc=0,column='Metric',value=metrices)
print(second_Models_results)

