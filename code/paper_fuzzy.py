
'''
Projeto Disciplina Aprendizagem de Maquina
Atividade 2 parte II - KNN 

@claudioalvesmonteiro
'''

# import packages
import pandas as pd 
import numpy as np

# import data
df = pd.read_csv('data/preprocessed_mfeat.csv')


#==================================
# calculate dissimilarity matrix
#==================================


def select_view(data, view):
    ''' function to select view data
    '''
    cols = [x for x in data.columns if view in x ]
    return data[cols]


def dissimilarityMatrix(data):
    ''' function to calculate dissimilarity matrix of dataframe
        based on euclidian distance
    '''
    diss_matrix = {}
    for col in data.columns:
        diss_matrix[col] = []
        for col2 in data.columns:
            distance = np.linalg.norm(df[col]-df[col2]) 
            diss_matrix[col].append(distance)
    return pd.DataFrame(diss_matrix)


diss_fac = dissimilarityMatrix(select_view(df, 'view_fac'))
diss_fou = dissimilarityMatrix(select_view(df, 'view_fou'))
diss_kar = dissimilarityMatrix(select_view(df, 'view_kar'))