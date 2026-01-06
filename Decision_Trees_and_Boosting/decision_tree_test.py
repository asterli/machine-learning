import decision_tree
from boosting import AdaBoost
from sklearn.metrics import accuracy_score
from decision_tree import check_behavior, train_model

from sklearn.ensemble import AdaBoostClassifier

import numpy as np
import json

# Hand-written digits data
def data_loader_mnist(dataset='mnist_subset.json'):
  # This function reads the MNIST data and separate it into train, val, and test set
  with open(dataset, 'r') as f:
        data_set = json.load(f)
  train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']
  train_y = np.array(train_set[1])
  test_y = np.array(test_set[1])
  train_y[train_y < 5] = 0
  train_y[train_y >= 5] = 1
  test_y[test_y < 5] = 0
  test_y[test_y >= 5] = 1
  
  n_samples = 10000
  n_te_samples = 2000

  #flatten the images into 1D inputs for tree-based methods
  train_X = np.array(train_set[0]).reshape((n_samples//2, -1)) 
  test_X = np.array(test_set[0]).reshape((n_te_samples//2, -1))

  return np.asarray(train_X), \
          np.asarray(test_X), \
          np.asarray(train_y), \
          np.asarray(test_y)
        

# Examine effect of max_depth
def max_depth_testing():
    name = 'max_depth'
    vals = [1, 3, 5, 10, 15, 20]
    X_tr, X_te, y_tr, y_te = data_loader_mnist('mnist_subset.json')
    
    model_type = 'DT'
    check_behavior(X_tr, X_te, y_tr, y_te, vals, name, model_type, ylim = 0.7)
    print('[success] : max depth evaluation on Decision Tree')

    model_type = 'RF'
    check_behavior(X_tr, X_te, y_tr, y_te, vals, name, model_type, ylim = 0.7)
    print('[success] : max depth evaluation on Random Forest')
    
    model_type = 'AdaBoost'
    check_behavior(X_tr, X_te, y_tr, y_te, vals, name, model_type, ylim = 0.7)
    print('[success] : max depth evaluation on AdaBoost')

# Examine effect of n_estimators
def n_estimaters_testing():
    name = 'n_estimators'
    vals = [1, 10, 25, 50, 75, 100, 200]
    X_tr, X_te, y_tr, y_te = data_loader_mnist('mnist_subset.json')
    
    model_type = 'RF'
    check_behavior(X_tr, X_te, y_tr, y_te, vals, name, model_type)
    print('[success] : n estimators evaluation on Random Forest')
    
    model_type = 'AdaBoost'
    check_behavior(X_tr, X_te, y_tr, y_te, vals, name, model_type)
    print('[success] : n estimators evaluation on AdaBoost')
    

if __name__ == '__main__':
    
    print("\n======== Examine effect of max_depth ========")
    max_depth_testing()
    
    print("\n======== Examine effect of n_estimators ========")
    n_estimaters_testing()
    
    