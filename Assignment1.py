"""
Creation date:  Fri Feb 21 23:32:36 2020
Authors:        Kalman Bogdan (s182210) and Bertalan Kovacs (182596)
Description:    Mandatory assignment for Machine Learning course
"""

# We have added an extra column of the "category" based on the grade of the real estate:
# Low, Medium, High. It represents the quality of contruction and design. It will be used for
# classification. Based on that, an other column called "encoded" has also been implemented 
# where the correspondance is the following: Low - 1, Medium - 2, High - 0

#%% Data import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import svd
from scipy import stats

filename = '../Data/kc_house_data.csv'
df = pd.read_csv(filename)
raw_data = df.get_values()

cols = range(0, 23) 
X = raw_data[:, cols]
attributeNames = np.asarray(df.columns[cols])
classLabels = raw_data[:,22]
classNames = np.unique(classLabels)
N, M = X.shape
C = len(classNames)

#%% Modifying the dataset
# Add year and month columns
X_date = np.asarray(X[:,1])
#X_yrRenovated = np.asarray(X[:,15])
X_year = np.zeros((N,1))
X_month = np.zeros((N,1))
#X_isRenovated = np.zeros((N,1))
for i in range(len(X_date)):
    str_tmp = X_date[i]
    X_year[i,0] = float(str_tmp[0:4])
    X_month[i,0] = float(str_tmp[4:6])
    #if float(X_yrRenovated[i]) > 0:
        #X_isRenovated[i,0] = 0
    #else:
        #X_isRenovated[i,0] = 1

# Now we add the two new data coloumns to the already existing structure
X = np.concatenate((X[:,0:2], X_year[:,:], X_month[:,:], X[:,2:23]), axis=1)
attributeNames = np.hstack((attributeNames[0:2], 'year', 'month', attributeNames[2:23]))

N, M = X.shape

#%% Classification problem
X_c = X.copy();
attributeNames_c = attributeNames.copy();

i = 7; j = 4;
color = ['r','g','b']
plt.title('Real estate classification problem')
for c in range(len(classNames)):
    idx = X_c[:,23] == c
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=10, alpha=0.5,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.savefig('Classification_problem.png')
plt.show()

#%% Regression problem
# We can think about the price of the real estate as a function of it's different
# properties like living space, number of bedrooms or view
i = 21; j = 4;
plt.title('Regression - price based on squarefeet lot 15')
plt.scatter(x=X_c[:, i], y=X_c[:, j], s=5, alpha=0.5, label=classNames[c])
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.savefig('Regression_problem.png')
plt.show()

#%% Missing values, corrupted data, standard statistics
# No missing data or corrupted values

X_means = X[:,2:24].mean(0)
X_mode = stats.mode(X[:,2:24])
X_range = np.zeros(24)
X_stddev = np.zeros(24)
for i in np.arange(2, 24):
    X_range[i] = np.max(X[:,i]) - np.min(X[:,i])
    X_stddev[i] = X[:,i].std(0)

#%% PCA - normalized vs original - We did not use this at the end, we were just
# curious how much normalization effects our results 
# Only take relevant data
X_rel = np.zeros((N, M))
for i in np.arange(2, 24):
    X_rel[:, i] = np.asarray(X[:, i])
X_rel = X_rel[:,2:24]  
# Subtract mean value from data
Y1 = X_rel - np.ones((N, 1))*X_rel.mean(0)

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset
Y2 = X_rel - np.ones((N, 1))*X_rel.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
Ys = [Y1, Y2]
titles = ['Zero-mean', 'Zero-mean and unit variance']
threshold = 0.9
# PCs to plot (the projection)
i = 0
j = 1

plt.figure(figsize=(10,15))
plt.subplots_adjust(hspace=.4)
nrows=3
ncols=2
for k in range(2):
    U,S,Vh = svd(Ys[k],full_matrices=False)
    V=Vh.T 
    if k==1: V = -V; U = -U; 
    
    rho = (S*S) / (S*S).sum() 
    
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[X[:,23]==c,i], Z[X[:,23]==c,j], '.', alpha=.5)
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.title(titles[k] + '\n' + 'Projection' )
    plt.legend(classNames)
    plt.axis('equal')
    
    # Plot attribute coefficients in principal component space
    plt.subplot(nrows, ncols, 3+k)
    for att in range(V.shape[1]):
        plt.arrow(0,0, V[att,i], V[att,j])
        plt.text(V[att,i], V[att,j], attributeNames[att+2])
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.xlabel('PC'+str(i+1))
    plt.ylabel('PC'+str(j+1))
    plt.grid()
    # Add a unit circle
    plt.plot(np.cos(np.arange(0, 2*np.pi, 0.01)), 
         np.sin(np.arange(0, 2*np.pi, 0.01)));
    plt.title(titles[k] +'\n'+'Attribute coefficients')
    plt.axis('equal')
            
    # Plot cumulative variance explained
    plt.subplot(nrows, ncols,  5+k);
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.title(titles[k]+'\n'+'Variance explained')
plt.show()

#%% Once we compared the two, we make the plot based on the normalised values

plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.savefig('PCA_variance.png')
plt.show()

#%% Then we check each attribute's correspondence to PC directions

U,S,Vh = svd(Y2,full_matrices=False)
V=Vh.T
attributeNames_PCA = attributeNames[2:25]
pcs = [0,1]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .3
r = np.arange(1,M+1-3)
for i in pcs:   
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames_PCA)
plt.xticks(rotation=-90)
plt.ylim(top=0.4, bottom=-0.6)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('PCA component coefficients')
plt.savefig('PCA_coefficients.png')
plt.show()

#%% The we plot the relation between the first 5 PCA components
i = 0
j = 0
plt.figure(figsize=(20,20))
plt.subplots_adjust(hspace=.15, wspace=.25)
plt.title('Scatter matrix of principal components')
plt.legend(classNames)
nrows=5
ncols=5
for k in range(25):
    rho = (S*S) / (S*S).sum() 
    
    Z = U*S;
    
    # Plot projection
    plt.subplot(nrows, ncols, 1+k)
    C = len(classNames)
    for c in range(C):
        plt.plot(Z[X[:,23]==c,i], Z[X[:,23]==c,j], '.', alpha=.5)
    plt.grid()
    
    j = j+1;
    if j == 5:
        j = 0
        i = i+1;
plt.savefig('PCA_scatter.png')
plt.show()
#%% Identifying outliers