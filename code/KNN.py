'''
Projeto Disciplina Aprendizagem de Maquina
Atividade 2 parte II - KNN 

@claudioalvesmonteiro
'''


# import packages
import pandas as pd
import numpy as np

# read data
fac = pd.read_csv('data/mfeat-fac', sep='\t', header=None)
fou = pd.read_csv('data/mfeat-fou', sep='\t', header=None)
kar = pd.read_csv('data/mfeat-kar', sep='\t', header=None)

#==================================
# preprocessing
#==================================

#------------- transform data to pandas df 
def multiFeaturesData(data, view):
    data_pd = {}
    for line in data[0]:
        new_line =[]
        split_line = line.split(' ')
        for i in split_line:
            if i != '':
                new_line.append(i)
        for i in range(len(new_line)):
            try:
                data_pd[(view+str(i))].append(new_line[i]) 
            except:
                data_pd[(view+str(i))] = []
                data_pd[(view+str(i))].append(new_line[i]) 
    return data_pd

# apply transformation for each view
fac = pd.DataFrame(multiFeaturesData(fac, 'view_fac_'))
fou = pd.DataFrame(multiFeaturesData(fou, 'view_fou_'))
kar = pd.DataFrame(multiFeaturesData(kar, 'view_kar_'))

# combine data
dataset = pd.concat([fac,fou,kar],axis=1)

# normalization of data
def normalize(dataset):
    for col in dataset.columns:
        dataset[col] = [float(x) for x in dataset[col]]
        med = dataset[col].mean()
        stdev = dataset[col].std()
        dataset[col] = [(x-med)/stdev for x in dataset[col] ]
    return dataset

dataset = normalize(dataset)

#---------- make target of data
target =[]
cont=1
for i in range(10):
    while cont <= 200:
        target.append(i)
        cont+=1 
    cont=1

dataset['target'] = target

# shuffle data and make k fold
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset['kfold'] = target

# divide
fold = 0

features_train = dataset[dataset['kfold']!=fold].drop(['target', 'kfold'], axis=1)
target_train = dataset['target'][dataset['kfold']!=fold]
features_test = dataset[dataset['kfold']==fold].drop(['target', 'kfold'], axis=1)
target_test = dataset['target'][dataset['kfold']==fold]


#===================
# KNN
#===================

def select_view(data, view):
    cols = [x for x in data.columns if view in x ]
    return data[cols]


def KnnViewModelling(features_train, target_train, features_test, target_test):

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


#### 30 times 10 K-FOLD EXPERIMENT
experiments_results = []
for fold in dataset['kfold'].unique():
    print(fold)
    cont=1
    while cont <= 30:
        features_train = dataset[dataset['kfold']!=fold].drop(['target', 'kfold'], axis=1).reset_index(drop=True)
        target_train = dataset['target'][dataset['kfold']!=fold].reset_index(drop=True)
        features_test = dataset[dataset['kfold']==fold].drop(['target', 'kfold'], axis=1).reset_index(drop=True)
        target_test = dataset['target'][dataset['kfold']==fold].reset_index(drop=True)
        acc, preds = KnnViewModelling(features_train, target_train, features_test, target_test)
        experiments_results.append(acc)
        cont+=1
    print(experiments_results)

#### save experiment data
results_data = pd.DataFrame({'results': experiments_results})
results_data.to_csv('results_knn_experiment.csv')

results_data['results'].mean()