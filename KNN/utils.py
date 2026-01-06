import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    tp = fp = fn = 0
    for r, p in zip(real_labels, predicted_labels):
        if r == 1 and p == 1:
            tp += 1
        elif r == 0 and p == 1:
            fp += 1
        elif r == 1 and p == 0:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return (2*precision*recall)/(precision+recall)


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        md = np.power(np.sum(np.abs(np.array(point1)-np.array(point2))**3),1/3)
        return md

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        ed = np.sqrt(np.sum((np.array(point1)-np.array(point2))**2))
        return ed

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        p1=np.array(point1)
        p2=np.array(point2)
        norm1 = np.linalg.norm(p1)
        norm2=np.linalg.norm(p2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        csd = 1-np.dot(p1,p2)/(norm1*norm2)
        return csd



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_f1 = -1
        
        for dist_name, dist_func in distance_funcs.items():
            for k in range(1,30,2):
                model = KNN(k, dist_func)
                model.train(x_train,y_train)
                preds = model.predict(x_val)
                score = f1_score(y_val,preds)
                if (
                    score > best_f1 or
                    (score == best_f1 and (self.best_distance_function is None or list(distance_funcs.keys()).index(dist_name) < list(distance_funcs.keys()).index(self.best_distance_function))) or
                    (score == best_f1 and dist_name == self.best_distance_function and k < self.best_k)
                ):
                    best_f1=score
                    self.best_k = k
                    self.best_distance_function = dist_name
                    self.best_model = model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        best_f1 = -1
        
        for scaler_name, scaler_class in scaling_classes.items():
            scaler = scaler_class()
            x_train_scaled = scaler(x_train)
            x_val_scaled = scaler(x_val)
            for dist_name, dist_func in distance_funcs.items():
                for k in range(1,30,2):
                    model = KNN(k, dist_func)
                    model.train(x_train_scaled,y_train)
                    preds = model.predict(x_val_scaled)
                    score = f1_score(y_val,preds)
                    if (
                        score > best_f1 or
                        (score == best_f1 and list(scaling_classes.keys()).index(scaler_name) < list(scaling_classes.keys()).index(self.best_scaler)) or
                        (score == best_f1 and scaler_name == self.best_scaler and list(distance_funcs.keys()).index(dist_name) < list(distance_funcs.keys()).index(self.best_distance_function)) or
                        (score == best_f1 and scaler_name == self.best_scaler and dist_name == self.best_distance_function and k < self.best_k)
                     ):
                        best_f1=score
                        self.best_k = k
                        self.best_distance_function = dist_name
                        self.best_model = model
                        self.best_scaler = scaler_name


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalized = []
        for f in features:
            norm = np.linalg.norm(f)
            if norm == 0:
                normalized.append(list(f))
            else:
                normalized.append(list(np.array(f)/norm))
        return normalized


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        feats = np.array(features)
        mins = feats.min(axis=0)
        maxs = feats.max(axis=0)
        diff = maxs-mins
        diff[diff==0]=1
        scaled=(feats-mins)/diff
        return scaled.tolist()
