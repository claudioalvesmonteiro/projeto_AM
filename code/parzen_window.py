



# https://stats.stackexchange.com/questions/186269/probabilistic-classification-using-kernel-density-estimation


# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work

# import packages
import pandas as pd

# import data
df = pd.read_csv('data/preprocessed_mfeat.csv')

# 
df.head()




#=============================================
# PARZEN WINDOW [kernel desnity estimation]
#=============================================

import numpy as np
from sklearn.neighbors.kde import KernelDensity
from matplotlib import pyplot as plt
sp = 0.01

samples = np.random.uniform(0,1,size=(50,2))  # random samples
x = y = np.linspace(0,1,100)
X,Y = np.meshgrid(x,y)     # creating grid of data , to evaluate estimated density on


from sklearn.neighbors import KernelDensity

#def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
#    """Build 2D kernel density estimate (KDE)."""

# create grid of sample locations (default: 100x100)
xbins=100j; ybins=100j

xx, yy = np.mgrid[x.min():x.max():xbins, y.min():y.max():ybins]

xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
xy_train  = np.vstack([y, x]).T

kde_skl = KernelDensity(bandwidth=0.2)
kde_skl.fit(xy_train)

# score_samples() returns the log-likelihood of the samples
np.exp(kde_skl.score_samples(xy_sample))


### SEPA

# construir um KDE pra cada classe

# retorna o KDE prob para aquela classe

# combinar os KDEs para retornar classe com maior valor 



def select_view(data, view):
    ''' function to select view data
    '''
    cols = [x for x in data.columns if view in x ]
    return data[cols]


def KnnViewModelling(features_train, target_train, features_test, target_test):
    ''' build and run models for each view,
        combine probabilities of models for the final
        decision, return accuracy of model  
    '''

    # build models for each view and return probabilities
    from sklearn.neighbors import KNeighborsClassifier
    knn_view1 = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn_view1.fit(select_view(features_train, 'view_fac'), target_train)
    pred_knn_view1 = list(knn_view1.predict_proba(select_view(features_test,  'view_fac')))

    knn_view2 = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn_view2.fit(select_view(features_train, 'view_fou'), target_train)
    pred_knn_view2 = list(knn_view2.predict_proba(select_view(features_test,  'view_fou')))

    knn_view3 = KNeighborsClassifier(n_neighbors=3, weights='distance')
    knn_view3.fit(select_view(features_train, 'view_kar'), target_train)
    pred_knn_view3 = list(knn_view3.predict_proba(select_view(features_test,  'view_kar')))

    # probability combination
    predictions = []
    for i in range(len(features_test)):
        # sum probs
        sum_prob = pred_knn_view1[i]+pred_knn_view2[i]+pred_knn_view3[i]
        # normalize
        norm_prob = (sum_prob - sum_prob.min()) / (sum_prob - sum_prob.min()).sum()
        # decision
        decision = np.where(norm_prob == max(norm_prob))[0][0]
        predictions.append(decision)

    # evaluate decision
    correct_shots = [1 if predictions[i] ==  target_test[i] else 0 for i in range(len(target_test))]
    accuracy = sum(correct_shots)/len(correct_shots)
    
    return accuracy, predictions
