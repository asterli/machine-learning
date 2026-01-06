import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Part 2: Implementation of AdaBoost with decision trees as weak learners

class AdaBoost:
  def __init__(self, n_estimators=60, max_depth=10):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.betas = []
    self.models = []
    
  def fit(self, X, y):
    ###########################TODO#############################################
    # In this part, please implement the adaboost fitting process based on the 
    # lecture and update self.betas and self.models, using decision trees with 
    # the given max_depth as weak learners

    # Inputs: X, y are the training examples and corresponding (binary) labels
    
    # Hint 1: remember to convert labels from {0,1} to {-1,1}
    # Hint 2: DecisionTreeClassifier supports fitting with a weighted training set
    y_sign = np.where(y == 0, -1, 1)
    N= X.shape[0]
    w = np.ones(N)/N
    self.models = []
    self.betas = []
    for _ in range(self.n_estimators):
        model = DecisionTreeClassifier(max_depth = self.max_depth)
        model.fit(X, y_sign, sample_weight=w)
        preds = model.predict(X)
        err = np.sum(w*(preds!=y_sign))
        err=max(err,1e-10)
        beta = 0.5 *np.log((1-err)/err)
        w *= np.exp(-beta*y_sign*preds)
        w/=np.sum(w)
        self.models.append(model)
        self.betas.append(beta)
        
    return self
    
  def predict(self, X):
    ###########################TODO#############################################
    # In this part, make prediction on X using the learned ensemble
    # Note that the prediction needs to be binary, that is, 0 or 1.
    temp = np.zeros(X.shape[0])
    for beta,model in zip(self.betas,self.models):
        temp += beta * model.predict(X)
    preds = np.where(temp>=0,1,0)
    return preds
    
  def score(self, X, y):
    accuracy = accuracy_score(y, self.predict(X))
    return accuracy

