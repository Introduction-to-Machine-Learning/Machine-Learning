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
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,22]
classNames = np.unique(classLabels)
N, M = X.shape

#%% REGRESSION A.1: Feature transformation

# Add year, month columns and convert renovation year to true/false as integer
X_date = np.asarray(X[:,1])
X_yrRenovated = np.asarray(X[:,15])
X_year = np.zeros((N,1))
X_month = np.zeros((N,1))
X_isRenovated = np.zeros((N,1))
for i in range(len(X_date)):
    str_tmp = X_date[i]
    X_year[i,0] = float(str_tmp[0:4])
    X_month[i,0] = float(str_tmp[4:6])
    if float(X_yrRenovated[i]) > 0:
        X_isRenovated[i,0] = 0
    else:
        X_isRenovated[i,0] = 1

# Now we add the three new data coloumns to the already existing structure
Y = X[:,2]
X = np.concatenate((X_year[:,:], X_month[:,:], X_isRenovated[:,:], X[:,3:15], X[:,16:21]), axis=1)
attributeNames = np.hstack(('year', 'month', 'is_renovated', attributeNames[3:15], attributeNames[16:21]))

N, M = X.shape

# Standardization 
X_rel = np.array(X, dtype=np.float64)
X_centered = X_rel - np.ones((N, 1))*X_rel.mean(0)
X_norm = X_centered*(1/np.std(X_centered,0))


Y_centered = Y - Y.mean()
Y_norm = Y_centered/np.std(Y)


#%% REGRESSION A.2: Regularization by 1-layer crossvalidation 

# Regularization
X = X_norm
y = np.array(Y_norm, dtype=np.float64)
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = np.insert(attributeNames, 0, u'Offset')
M = M+1

# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,np.linspace(0,4,25,endpoint=False))

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
    
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
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
plt.figure(1, figsize=(12,8))
plt.subplot(1,2,1)
plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
plt.xlabel('Regularization factor')
plt.ylabel('Mean Coefficient Values')
plt.grid()
# You can choose to display the legend, but it's omitted for a cleaner 
# plot, since there are many attributes
#legend(attributeNames[1:], loc='best')

plt.subplot(1,2,2)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Squared error (crossvalidation)')
plt.legend(['Train error','Validation error'])
plt.grid()

plt.show()

#%% REGRESSION A.3: Selected attributes for future prediction
w_opt = np.squeeze(np.mean(w[:,:,np.where(lambdas == opt_lambda)],axis=1))
w_initial = np.squeeze(np.mean(w_noreg,axis=1))

# create plot
fig, ax = plt.subplots(figsize=(8,6))
index = np.arange(M-1)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, w_initial[1:], bar_width,
alpha=opacity,
color='b',
label='Without regularization')

rects2 = plt.bar(index + bar_width, w_opt[1:], bar_width,
alpha=opacity,
color='g',
label='With optimal regularization')

plt.xlabel('Feature name')
plt.ylabel('Weight score')
plt.title('Weight score comparison to predict "price" by linear regression')
plt.xticks(index + bar_width, attributeNames[1:], rotation=90)
plt.legend()

plt.tight_layout()
plt.show()


#%% REGRESSION B.1: 2-layer crossvalidation for the three models: 
# Linear regression, ANN, Baseline

# Create crossvalidation partition for evaluation
K = 4
I = 2
CV_1 = model_selection.KFold(K, shuffle=True)
CV_2 = model_selection.KFold(I, shuffle=True)

# Values of lambda and hidden layers
lambdas = np.power(10.,np.linspace(0,4,25,endpoint=False))
n_hidden_units_list = [1, 4, 8, 12]

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
Error_test_ANN = np.empty((K,1))
optimal_layer_num = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))


for (k, (train_index, test_index)) in enumerate(CV_1.split(X,y)): 
    print('\nCrossvalidation fold layer-1: {0}/{1}'.format(k+1,K))    
 
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, I)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K-1:
        plt.figure(k, figsize=(12,8))
        plt.subplot(1,2,1)
        plt.semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        plt.subplot(1,2,2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        plt.loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.legend(['Train error','Validation error'])
        plt.grid()
    
    # --------------------------------- ANN start ---------------------------------
    # Parameters for neural network classifier
    n_replicates, max_iter = 1, 10000  # number of hidden units, networks and max iteration
    
    # Setup figure for display of learning curves and error rates in fold
    summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
                  'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
    
    val_errors = [] # make a list for storing generalizaition error in each loop
    opt_layer_nums = []
    for (i, (train_index_ANN, test_index_ANN)) in enumerate(CV_2.split(X_train,y_train)): 
        print('\n\tCrossvalidation fold layer-2: {0}/{1}'.format(i+1,I))    
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train_ANN = torch.Tensor(X_train[train_index_ANN,:])
        y_train_ANN = torch.Tensor(y_train[train_index_ANN])
        X_test_ANN = torch.Tensor(X_train[test_index_ANN,:])
        y_test_ANN = torch.Tensor(y_train[test_index_ANN])
        
        learning_curves = []
        layer_errors = []
        nets = []
        
        for h, n_hidden_units in enumerate(n_hidden_units_list):
            print('\t\tNumber of layers: {0}'.format(n_hidden_units)) 
            # Define the model
            model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                            torch.nn.Tanh(),   # 1st transfer function,
                            torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_ANN,
                                                           y=np.reshape(y_train_ANN, (-1,1)),
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
            
            # Determine estimated class labels for test set
            y_test_est_ANN = net(X_test_ANN)
            
            # Determine errors and errors
            se = (y_test_est_ANN.float().view(-1)-y_test_ANN.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean
            layer_errors.append(mse) # store error rate for current CV fold 
            learning_curves.append(learning_curve)
            nets.append(net)
            print('\n\t\tBest training loss: {:.6f}'.format(final_loss))
            print('\t\tBest validation loss: {:.6f}\n'.format(mse))
        
        opt_val_err = min(layer_errors)
        opt_layer_num = n_hidden_units_list[layer_errors.index(opt_val_err)]
        opt_learning_curve = learning_curves[layer_errors.index(opt_val_err)]
        opt_net = nets[layer_errors.index(opt_val_err)]
        opt_layer_nums.append(opt_layer_num)
        val_errors.append(opt_val_err)
        
        print("\t\tLowest error is {} with layer number {}.".format(opt_val_err, opt_layer_num))
    
        
        # Display the learning curve for the best net in the current fold      
        h, = summaries_axes[0].plot(opt_learning_curve, color=color_list[i])
        h.set_label('CV fold {0}'.format(i+1))
        summaries_axes[0].set_xlabel('Iterations')
        summaries_axes[0].set_xlim((0, max_iter))
        summaries_axes[0].set_ylabel('Loss')
        summaries_axes[0].set_title('Learning curves')
            
    # Display the MSE across folds
    summaries_axes[1].bar(np.arange(1, I+1), np.squeeze(np.asarray(val_errors)), color=color_list)
    summaries_axes[1].set_xlabel('Fold')
    summaries_axes[1].set_xticks(np.arange(1, I+1))
    summaries_axes[1].set_ylabel('MSE')
    summaries_axes[1].set_title('Test mean-squared-error')
        
    # print('Diagram of best neural net in last fold:')
    weights = [net[j].weight.data.numpy().T for j in [0,2]]
    biases = [net[j].bias.data.numpy() for j in [0,2]]
    tf =  [str(net[j]) for j in [1,2]]
    draw_neural_net(weights, biases, tf, attribute_names=list(attributeNames))
    
    # When dealing with regression outputs, a simple way of looking at the quality
    # of predictions visually is by plotting the estimated value as a function of 
    # the true/known value - these values should all be along a straight line "y=x", 
    # and if the points are above the line, the model overestimates, whereas if the
    # points are below the y=x line, then the model underestimates the value
    plt.figure(figsize=(10,10))
    y_est_ANN = y_test_est_ANN.data.view(-1).numpy(); y_true = y_test_ANN.data.numpy()
    axis_range = [np.min([y_est_ANN, y_true])-1,np.max([y_est_ANN, y_true])+1]
    plt.plot(axis_range,axis_range,'k--')
    plt.plot(y_true, y_est_ANN,'ob',alpha=.25)
    plt.legend(['Perfect estimation','Model estimations'])
    plt.title('Appartement price: estimated versus true value (for last CV-fold)')
    plt.ylim(axis_range); plt.xlim(axis_range)
    plt.xlabel('True value')
    plt.ylabel('Estimated value')
    plt.grid()
    plt.show()
    
    Error_test_ANN[k] = round(np.sqrt(np.mean(val_errors)), 4)
    optimal_layer_num[k] = round(np.median(opt_layer_nums))
    # ---------------------------------- ANN end ----------------------------------
    
    # To inspect the used indices, use these print statements
    print('\n\tCross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))
    print('\tTest error with Baseline model: ', Error_test_nofeatures[k])
    print('\tTest error with regression model: ', Error_test_nofeatures[k])
    print('\tTest error with regularizedregression model: ', Error_test_nofeatures[k])
    print('\tTest error with ANN, RMSE: {0}'.format(round(np.sqrt(np.mean(val_errors)), 4)))

plt.show()
# Display results
print('\n\nLinear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))

print('ANN:')
print('- Test error:     {0}'.format(Error_test_ANN.mean()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_ANN.sum())/Error_test_nofeatures.sum()))
