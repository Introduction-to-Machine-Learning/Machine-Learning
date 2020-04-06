# -*- coding: utf-8 -*-
"""
Creation date:  Fri Mar 30 14:32:36 2020
Authors:        Kalman Bogdan (s182210) and Bertalan Kovacs (182596)
Description:    Mandatory assignment 2 for Machine Learning course
"""
#%% Data import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import sklearn.linear_model
import sklearn.tree
from toolbox_02450 import jeffrey_interval, mcnemar

filename = 'C:/Users/Kalci/Documents/GitHub/Machine-Learning/kc_house_data.csv'
df = pd.read_csv(filename)
raw_data = df.to_numpy()
cols = range(0, 23) 
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,22]
classNames = np.unique(classLabels)
N, M = X.shape
C = len(classNames)

# Modifying the dataset
X_date = np.asarray(X[:,1])
X_year = np.zeros((N,1))
X_month = np.zeros((N,1))
for i in range(len(X_date)):
    str_tmp = X_date[i]
    X_year[i,0] = float(str_tmp[0:4])
    X_month[i,0] = float(str_tmp[4:6])
X = np.concatenate((X[:,0:2], X_year[:,:], X_month[:,:], X[:,2:23]), axis=1)
attributeNames = np.hstack((attributeNames[0:2], 'year', 'month', attributeNames[2:23]))
N, M = X.shape

# Standardization 
X_rel = np.zeros((N, M))
for i in np.arange(2, 24):
    X_rel[:, i] = np.asarray(X[:, i])
X_rel = X_rel[:,2:24]  
X_norm = X_rel - np.ones((N, 1))*X_rel.mean(0)
X_norm = X_norm*(1/np.std(X_norm,0))
attributeNames_norm = attributeNames[2:24]

X_means = X_norm[:,:].mean(0)
X_stddev = np.zeros(22)
for i in np.arange(0, 22):
    X_stddev[i] = X_norm[:,i].std(0)  
    
# Discard grade and encoded columns from the normalized model
X_norm = np.concatenate((X_norm[:,0:11], X_norm[:,12:21]), axis=1)
attributeNames_norm = np.hstack((attributeNames_norm[0:11], attributeNames_norm[12:21]))
N_norm, M_nor = X_norm.shape  
y=X[:,23];    
y=y.astype('int')

#%% Generating the baseline model
no_low=0; no_average=0; no_good=0; no_high=0; i=0;
for i in range (N):
    if X[i,23]==0:
        no_low += 1;
    elif X[i,23]==1:
        no_average += 1;
    elif X[i,23]==2:
        no_good += 1;
    else:
        no_high += 1;
common_value = np.argmax(np.array([no_low, no_average, no_good, no_high]))

K = 10;
CV = skl.model_selection.KFold(n_splits=K,shuffle=True)

error_test = np.ones(K);  
k=0;

for train_index, test_index in CV.split(X_norm):
    print('Computing external CV fold: {0}/{1}..'.format(k+1,K))
    X_train, y_train = X_norm[train_index,:], y[train_index]
    X_test, y_test = X_norm[test_index,:], y[test_index]
    i=0; missclass_rate = 0;
    for i in range(len(y_test)):
        if y_test[i] != common_value:
            missclass_rate = missclass_rate + 1;
    error_test[k] = missclass_rate/len(y_test);
    k+=1

#%% Visualization
names = ['$F_1$', '$F_2$', '$F_3$', '$F_4$', '$F_5$', '$F_6$', '$F_7$', '$F_8$', '$F_9$', '$F_{10}$']
f = plt.figure()
plt.plot(names[0:K], error_test*100)
plt.ylabel('Test error rate [%]')
plt.xlabel('Fold index $F_i$')
plt.title('Generalization error for baseline model')
plt.grid()
plt.show()

print('The estimate of the generalization error is: ',np.average(error_test)*100);
#%% Tree complexity parameter - constraint on maximum depth
tc = np.arange(3, 19, 1)

# K-fold crossvalidation
K_ext = 10;
CV_ext = skl.model_selection.KFold(n_splits=K_ext,shuffle=True)

# Initialize variable
K_int = 10;
Error_train_int = np.empty((len(tc),K_int))
Error_test_int = np.empty((len(tc),K_int))

Error_train_ext = np.zeros(K_ext);
Error_test_ext = np.ones(K_ext);

best_treedepth_int = np.zeros(K_ext);

k=0

for train_index_ext, test_index_ext in CV_ext.split(X_norm):
    print('Computing external CV fold: {0}/{1}..'.format(k+1,K_ext))

    # extract training and test set for current CV fold
    X_train_ext, y_train_ext = X_norm[train_index_ext,:], y[train_index_ext]
    X_test_ext, y_test_ext = X_norm[test_index_ext,:], y[test_index_ext]

    j=0;
    CV_int = skl.model_selection.KFold(n_splits=K_int,shuffle=True)
    
    for train_index_int, test_index_int in CV_int.split(X_train_ext):
        print('Computing internal CV fold: {0}/{1}..'.format(j+1,K_int))

        # extract training and test set for current CV fold
        X_train_int, y_train_int = X_train_ext[train_index_int,:], y_train_ext[train_index_int]
        X_test_int, y_test_int = X_train_ext[test_index_int,:], y_train_ext[test_index_int]

        for i, t in enumerate(tc):
            # Fit decision tree classifier, Gini split criterion, different pruning levels
            dtc = skl.tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
            dtc = dtc.fit(X_train_int,y_train_int.ravel())
            y_est_test = dtc.predict(X_test_int)
            y_est_train = dtc.predict(X_train_int)
            # Evaluate misclassification rate over train/test data (in this CV fold)
            misclass_rate_test = np.sum(y_est_test != y_test_int) / float(len(y_est_test))
            misclass_rate_train = np.sum(y_est_train != y_train_int) / float(len(y_est_train))
            Error_test_int[i,j], Error_train_int[i,j] = misclass_rate_test, misclass_rate_train
        j+=1;
    
    
    f = plt.figure()
    plt.boxplot(Error_test_int.T,positions=tc)
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Test error across CV folds, K={0})'.format(K_int))
    
    f = plt.figure()
    plt.plot(tc, Error_train_int.mean(1))
    plt.plot(tc, Error_test_int.mean(1))
    plt.xlabel('Model complexity (max tree depth)')
    plt.ylabel('Error (misclassification rate, CV K={0})'.format(K_int))
    plt.legend(['Error_train','Error_test'])
    plt.show()
    
    best_treedepth_int[k] = tc[np.argmin(np.average(Error_test_int, axis = 1))]
    print('The optimal tree depth is: ', best_treedepth_int[k]);
    dtc = skl.tree.DecisionTreeClassifier(criterion='gini', max_depth=best_treedepth_int[k])
    dtc = dtc.fit(X_train_ext,y_train_ext.ravel())
    y_est_test = dtc.predict(X_test_ext)
    y_est_train = dtc.predict(X_train_ext)
    misclass_rate_test = np.sum(y_est_test != y_test_ext) / float(len(y_est_test))
    misclass_rate_train = np.sum(y_est_train != y_train_ext) / float(len(y_est_train))
    Error_test_ext[k], Error_train_ext[k] = misclass_rate_test, misclass_rate_train
    k+=1;

print('Done with classification tree')
#%% Visualization
names = ['$M_1$', '$M_2$', '$M_3$', '$M_4$', '$M_5$', '$M_6$', '$M_7$', '$M_8$', '$M_9$', '$M_{10}$']
f = plt.figure()
plt.plot(names[0:K_ext], Error_test_ext*100, Error_train_ext*100)
plt.ylabel('Error rate [%]')
plt.xlabel('Model index $M_i$')
plt.title('Training error and estimate of generalization error ')
plt.legend(['Test error','Train error'])
plt.grid()
plt.show()

best_treedepth = best_treedepth_int[np.argmin(Error_test_ext)]
print('The estimate of the generalization error is: ',np.average(Error_test_ext)*100);
print('The tree depth in the best case is:', best_treedepth, 'with the error of', np.min(Error_test_ext)*100, '%');

#%% Logistic regression
K_ext = 10;
K_int=10;
lambda_interval = np.logspace(-2, 4, 50)
CV_ext = skl.model_selection.KFold(n_splits=K_ext,shuffle=True)
Error_train_int = np.zeros((K_int, len(lambda_interval)))
Error_test_int = np.zeros((K_int, len(lambda_interval)))
coefficient_norm = np.zeros((len(lambda_interval),K_int))
w_est = np.zeros((20,K_int))
min_error = np.zeros(K_int)
min_error_val = np.zeros(K_int)
best_error = np.zeros(K_ext)
best_lambda = np.zeros(K_ext)
opt_lambda_idx = np.zeros(K_int)
opt_lambda = np.zeros(K_int)
Error_train_ext = np.zeros(K_ext);
Error_test_ext = np.ones(K_ext);
k=0

for train_index_ext, test_index_ext in CV_ext.split(X_norm):
    print('Computing external CV fold: {0}/{1}..'.format(k+1,K_ext))

    X_train_ext, y_train_ext = X_norm[train_index_ext,:], y[train_index_ext]
    X_test_ext, y_test_ext = X_norm[test_index_ext,:], y[test_index_ext]

    j=0;
    CV_int = skl.model_selection.KFold(n_splits=K_int,shuffle=True)
    
    for train_index_int, test_index_int in CV_int.split(X_train_ext):
        print('Computing internal CV fold: {0}/{1}..'.format(j+1,K_int))
        
        X_train_int, y_train_int = X_train_ext[train_index_int,:], y_train_ext[train_index_int]
        X_test_int, y_test_int = X_train_ext[test_index_int,:], y_train_ext[test_index_int]

        for i in range(0, len(lambda_interval)):
            mdl = skl.linear_model.LogisticRegression(penalty='l2', max_iter=3000, multi_class='multinomial', C=1/lambda_interval[i])
            mdl.fit(X_train_int, y_train_int)
            
            y_train_est = mdl.predict(X_train_int).T
            y_test_est = mdl.predict(X_test_int).T
            
            Error_train_int[j,i] = np.sum(y_train_est != y_train_int) / len(y_train_int)
            Error_test_int[j,i] = np.sum(y_test_est != y_test_int) / len(y_test_int)
            
            w_est[:,j] = mdl.coef_[0]
            coefficient_norm[i,j] = np.sqrt(np.sum(w_est**2))
              
        min_error[j] = np.min(np.average(Error_test_int, axis = 1))
        min_error_val[j] = np.min(Error_test_int[j,:])
        opt_lambda_idx[j] = np.argmin(Error_test_int[j,:])
        opt_lambda[j] = lambda_interval[np.int_(opt_lambda_idx[j])]
        
        plt.figure(figsize=(8,8))
        plt.semilogx(np.transpose(lambda_interval), Error_train_int[j,:]*100)
        plt.semilogx(np.transpose(lambda_interval), Error_test_int[j,:]*100)
        plt.semilogx(opt_lambda[j], min_error_val[j]*100, 'o')
        plt.text(1e-2, 20, "Minimum test error: " + str(np.round(min_error_val[j]*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda[j]),2)))
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.ylabel('Error rate (%)')
        plt.title('Classification error')
        plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
        plt.grid()
        plt.show()    
        
        plt.figure(figsize=(8,8))
        plt.semilogx(lambda_interval, coefficient_norm[:,j],'k')
        plt.ylabel('L2 Norm')
        plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
        plt.title('Parameter vector L2 norm')
        plt.grid()
        plt.show()  
        j+=1;
    
    # So now we will have the best model and we can use that
    best_error[k] = np.min(min_error[:]);
    best_model = np.argmin(min_error[:]);
    best_lambda[k] = opt_lambda[best_model];
    mdl = skl.linear_model.LogisticRegression(penalty='l2', max_iter=3000, multi_class='multinomial', C=1/best_lambda[k])
    mdl.fit(X_train_ext, y_train_ext)
            
    y_train_est = mdl.predict(X_train_ext).T
    y_test_est = mdl.predict(X_test_ext).T
            
    Error_train_ext[k] = np.sum(y_train_est != y_train_ext) / len(y_train_ext)
    Error_test_ext[k] = np.sum(y_test_est != y_test_ext) / len(y_test_ext)
            
    w_est_ext = mdl.coef_[0]
    coefficient_norm_ext = np.sqrt(np.sum(w_est**2))

    k+=1;
    
#%% Visualization
names = ['$M_1$', '$M_2$', '$M_3$', '$M_4$', '$M_5$', '$M_6$', '$M_7$', '$M_8$', '$M_9$', '$M_{10}$']
f = plt.figure()
plt.plot(names[0:K_ext], Error_test_ext*100, Error_train_ext*100)
plt.ylabel('Error rate [%]')
plt.xlabel('Model index $M_i$')
plt.title('Training error and estimate of generalization error ')
plt.legend(['Test error','Train error'])
plt.grid()
plt.show()

print('The estimate of the generalization error is: ',np.average(Error_test_ext));
print('The best model on the outer fold is number', np.argmin(Error_test_ext)+1,'with lamda=',best_lambda[np.argmin(Error_test_ext)],'and error=',np.min(Error_test_ext))

#%% Statistical evaluations of results
K = 10;
CV = skl.model_selection.KFold(n_splits=K,shuffle=True)
k=0
y_true = []
y_hat_lr = []
y_hat_tree = []

for train_index, test_index in CV.split(X_norm):
    print('Computing external CV fold: {0}/{1}..'.format(k+1,K))
    X_train, y_train = X_norm[train_index,:], y[train_index]
    X_test, y_test = X_norm[test_index,:], y[test_index]
    
    mdl = skl.linear_model.LogisticRegression(penalty='l2', max_iter=3000, multi_class='multinomial', C = best_lambda[np.argmin(Error_test_ext)])
    mdl.fit(X_train, y_train)
    y_est_lr = mdl.predict(X_test).T
    
    dtc = skl.tree.DecisionTreeClassifier(criterion='gini', max_depth=best_treedepth)
    dtc = dtc.fit(X_train,y_train.ravel())
    y_est_tree = dtc.predict(X_test)
    
    y_hat_lr = np.concatenate((y_hat_lr, y_est_lr), axis=0)
    y_hat_tree = np.concatenate((y_hat_tree, y_est_tree), axis=0)
    y_true = np.concatenate((y_true, y_test), axis=0)
    
    k+=1;
    
# Jeffreys interval for baseline
y_hat_base = np.ones(len(y_true))*common_value;
alpha = 0.05;
[thetahatA, CIA] = jeffrey_interval(y_true, y_hat_base, alpha=alpha)
print("Theta point estimate", thetahatA, " CI: ", CIA)

# Jeffreys interval for CT
alpha = 0.05;
[thetahatA, CIA] = jeffrey_interval(y_true, y_hat_tree, alpha=alpha)
print("Theta point estimate", thetahatA, " CI: ", CIA)

# Jeffreys interval for logistic regression
alpha = 0.05;
[thetahatA, CIA] = jeffrey_interval(y_true, y_hat_lr, alpha=alpha)
print("Theta point estimate", thetahatA, " CI: ", CIA)

# Compare logistic regression and CT
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, y_hat_lr ,y_hat_tree, alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# Compare logistic regression and baseline
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, y_hat_lr, y_hat_base, alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

# Compare baseline and CT
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, y_hat_base, y_hat_tree, alpha=alpha)
print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)

#%%
names = attributeNames_norm
f = plt.figure()
plt.bar(names, w_est_ext, color=("blue"))
plt.ylabel('Coefficient value')
plt.xlabel('Attribute names')
plt.xticks(rotation=-90)
plt.title('Values of vector coefficients in classification')
plt.grid()
plt.show()

