# -*- coding: utf-8 -*-
"""
Creation date:  Mon Mar 30 17:50:36 2020
Authors:        Kalman Bogdan (s182210) and Bertalan Kovacs (182596)
Description:    Mandatory assignment for Machine Learning course Part 2
"""

# Data import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy import stats

import seaborn as sns
import missingno as msno
from IPython.display import display

import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net, draw_neural_net
import torch


filename = 'C:/Users/Kovacs Bertalan/Desktop/Intro_to_ML_Official/02450Toolbox_Python/Machine-Learning/kc_house_data.csv'
df = pd.read_csv(filename)
raw_data = df.get_values()

cols = range(0, 23) 
X_raw = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,22]
classNames = np.unique(classLabels)
N_raw, M_raw = X_raw.shape

#%% REGRESSION A.1: Feature transformation

# Add year, month columns and convert renovation year to true/false as integer
X_date = np.asarray(X_raw[:,1])
X_yrRenovated = np.asarray(X_raw[:,15])
X_year = np.zeros((N_raw,1))
X_month = np.zeros((N_raw,1))
X_isRenovated = np.zeros((N_raw,1))
for i in range(len(X_date)):
    str_tmp = X_date[i]
    X_year[i,0] = float(str_tmp[0:4])
    X_month[i,0] = float(str_tmp[4:6])
    if float(X_yrRenovated[i]) > 0:
        X_isRenovated[i,0] = 0
    else:
        X_isRenovated[i,0] = 1

# Now we add the three new data coloumns to the already existing structure
Y_proc = X_raw[:,2]
X_proc = np.concatenate((X_year[:,:], X_month[:,:], X_isRenovated[:,:], X_raw[:,3:15], X_raw[:,16:21]), axis=1)
attributeNames = np.hstack(('year', 'month', 'is_renovated', attributeNames[3:15], attributeNames[16:21]))

N_proc, M_proc = X_proc.shape

# Standardization 
X_rel = np.array(X_proc, dtype=np.float64)
X_centered = X_rel - np.ones((N_proc, 1))*X_rel.mean(0)
X_norm = X_centered*(1/np.std(X_rel,0))


Y_centered = Y_proc - Y_proc.mean()
Y_norm = Y_centered/np.std(Y_proc)


#%% REGRESSION A.2: Regularization by 1-layer crossvalidation 

# Regularization
X = X_norm
y = np.array(Y_norm, dtype=np.float64)
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.insert(attributeNames, 0, u'Offset')
M = M+1

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,np.linspace(0,4,25,endpoint=True))

# Regularized linear regression model validation based on rlr_validate()
# function from __init__.py
CV = model_selection.KFold(K, shuffle=True)
M = X.shape[1]
w = np.empty((M,K,len(lambdas)))
train_error = np.empty((K,len(lambdas)))
test_error = np.empty((K,len(lambdas)))
w_noreg = np.empty((M,K))

f = 0
y = y.squeeze()
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
        
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    w_noreg[:,f] = np.linalg.solve(XtX,Xty).squeeze()

    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

    f=f+1

opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))

# Display the results
# plt.figure(1, figsize=(12,8))
# plt.subplot(1,2,1)
# plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
# plt.xlabel('Regularization factor')
# plt.ylabel('Mean Coefficient Values')
# plt.grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

# plt.subplot(1,2,2)
# plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
# plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
# plt.xlabel('Regularization factor')
# plt.ylabel('Squared error (crossvalidation)')
# plt.legend(['Train error','Validation error'])
# plt.grid()

# plt.show()

#%% REGRESSION A.3: Selected attributes for future prediction
w_opt = np.squeeze(np.mean(w[:,:,np.where(lambdas == opt_lambda)],axis=1))
w_initial = np.squeeze(np.mean(w_noreg,axis=1))

# create plot
# fig, ax = plt.subplots(figsize=(8,6))
# index = np.arange(M-1)
# bar_width = 0.35
# opacity = 0.8

# rects1 = plt.bar(index, w_initial[1:], bar_width,
# alpha=opacity,
# color='b',
# label='Without regularization')

# rects2 = plt.bar(index + bar_width, w_opt[1:], bar_width,
# alpha=opacity,
# color='g',
# label='With optimal regularization')

# plt.xlabel('Feature name')
# plt.ylabel('Weight score')
# plt.title('Weight score comparison to predict "price" by linear regression')
# plt.xticks(index + bar_width, attributeNames[1:], rotation=90)
# plt.legend()

# plt.tight_layout()
# plt.show()


#%% REGRESSION B.1: 2-layer crossvalidation for the three models: 
# Linear regression, ANN, Baseline

# Create crossvalidation partition for evaluation
K1 = 10
K2 = 10
CV_1 = model_selection.KFold(K1, shuffle=True)
CV_2 = model_selection.KFold(K2, shuffle=True)

# Values of lambda and hidden layers
lambdas = np.power(10.,np.linspace(0,4,25,endpoint=False))
n_hidden_units_list = [1, 4, 7, 10, 11, 14, 17, 20]

# Initialize variables
#T = len(lambdas)
Error_prec = np.empty((K1,1))
Error_test = np.empty((K1,1))
Error_prec_rlr = np.empty((K1,1))
Error_test_rlr = np.empty((K1,1))
Error_prec_nofeatures = np.empty((K1,1))
Error_test_nofeatures = np.empty((K1,1))
Error_test_ANN = np.empty((K1,1))
optimal_lambda = np.empty((K1,1))
optimal_layer_num = np.empty((K1,1))
w_rlr = np.empty((M,K1))


for (k, (prec_index, test_index)) in enumerate(CV_1.split(X,y)): 
    print('\nCrossvalidation fold layer-1: {0}/{1}'.format(k+1,K1))    
 
    # extract training and test set for current CV fold
    X_prec = X[prec_index]
    y_prec = y[prec_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_prec, y_prec, lambdas, K2)

    Xty = X_prec.T @ y_prec
    XtX = X_prec.T @ X_prec
    
    # Compute mean squared error without using the input data at all
    Error_prec_nofeatures[k] = np.square(y_prec-y_prec.mean()).sum(axis=0)/y_prec.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_prec_rlr[k] = np.square(y_prec-X_prec @ w_rlr[:,k]).sum(axis=0)/y_prec.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    optimal_lambda[k] = opt_lambda
    
    # --------------------------------- ANN start ---------------------------------
    # Parameters for neural network classifier
    max_iter = 10000  # number of hidden units, networks and max iteration
    
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    val_errors = np.empty((K2,len(n_hidden_units_list))) # make a list for storing validation errors in each loop for every h
    for (i, (train_index_ANN, val_index_ANN)) in enumerate(CV_2.split(X_prec,y_prec)): 
        print('\n\tCrossvalidation fold layer-2: {0}/{1}'.format(i+1,K2))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train_ANN = torch.Tensor(X_train[train_index_ANN,1:])
        y_train_ANN = torch.Tensor(y_train[train_index_ANN])
        X_val_ANN = torch.Tensor(X_train[val_index_ANN,1:])
        y_val_ANN = torch.Tensor(y_train[val_index_ANN])
               
        for n, n_hidden_units in enumerate(n_hidden_units_list):
            print('\n\t\tNumber of layers: {0}'.format(n_hidden_units)) 
            # Define the model
            model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M-1, n_hidden_units), #M features to n_hidden_units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_ANN,
                                                           y=np.reshape(y_train_ANN, (-1,1)),
                                                           n_replicates=1,
                                                           max_iter=max_iter)
            
            # Determine estimated value for test set
            y_val_est_ANN = net(X_val_ANN)
            
            # Determine errors and errors
            se = (y_val_est_ANN.float().view(-1)-y_val_ANN.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_val_ANN)).data.numpy() #mean
            val_errors[i, n] = mse # store error rate for current CV fold 
            print('\t\tTaining loss: {:.6f}'.format(final_loss))
            print('\t\tValidation loss: {:.6f}'.format(mse))
            
    opt_val_err = np.min(np.mean(val_errors,axis=0))
    opt_layer_num = n_hidden_units_list[np.argmin(np.mean(val_errors,axis=0))]
    optimal_layer_num[k] = opt_layer_num

    X_prec_ANN = torch.Tensor(X_prec[:,1:])
    y_prec_ANN = torch.Tensor(y_prec)
    X_test_ANN = torch.Tensor(X_test[:,1:])
    y_test_ANN = torch.Tensor(y_test)

    # Run the training on the whole set with the optimal layer number:
    model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M-1, opt_layer_num), #M features to n_hidden_units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(opt_layer_num, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                   loss_fn,
                                                   X=X_prec_ANN,
                                                   y=np.reshape(y_prec_ANN, (-1,1)),
                                                   n_replicates=1,
                                                   max_iter=max_iter)
    
    # Determine estimated value for test set
    y_test_est = net(X_test_ANN)
    
    # Determine errors and errors
    se = (y_test_est.float().view(-1)-y_test_ANN.float())**2 # squared error
    Error_test_ANN[k] = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean     
    # ---------------------------------- ANN end ----------------------------------
    
    # To inspect the used indices, use these print statements
    print('\n\tCross validation fold {0}/{1}:'.format(k+1,K1))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))
    print('\tTest error with Baseline model: ', Error_test_nofeatures[k])
    print('\tTest error with regularized regression model: ', Error_test_rlr[k])
    print('\tTest error with ANN: ', Error_test_ANN[k])

# Display results
print('\n\nBaseline model:')
print('- Training error: {0}'.format(Error_prec_nofeatures.mean()))
print('- Test error:     {0}'.format(Error_test_nofeatures.mean()))
print('\n\nRegularized linear regression:')
print('- Training error: {0}'.format(Error_prec_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_prec_nofeatures.sum()-Error_prec_rlr.sum())/Error_prec_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))


print('ANN:')
print('- Test error:     {0}'.format(Error_test_ANN.mean()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_ANN.sum())/Error_test_nofeatures.sum()))

print('\n\nBaseline model errors: ', list(np.squeeze([error for error in Error_test_nofeatures])))
print('\nRegularized test errors: ', list(np.squeeze([error for error in Error_test_rlr])))
print('Optimal lambdas: ', list(np.squeeze(optimal_lambda)))
print('\nANN test errors: ', list(np.squeeze([error for error in Error_test_ANN])))
print('Optimal layer numbers: ', list(np.squeeze(optimal_layer_num)))