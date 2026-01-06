# **Decision Trees and Boosting (30 points)**  

## **Overview**  
In this assignment, you will explore decision trees and boosting methods, specifically focusing on AdaBoost. The assignment consists of three major parts:  

1. **Decision Trees and Random Forests** – Using existing implementations to understand fundamental concepts.  
2. **AdaBoost Implementation** – Implementing the AdaBoost algorithm from scratch using decision trees as base learners.  
3. **Examine the Effect of Different Parameters** – Test your implementation and examine the effect of different parameters.


## **Part 1: Decision Trees and Random Forests**  
Implmement decision trees and random forests in decision_tree.py by simply using scikit-learn; see [here](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).  

## **Part 2: Boosting (AdaBoost Implementation)**  
Implement AdaBoost from scratch in boosting.py. Complete the part that uses AdaBoost to train a model in decision_tree.py.

## **Part 3: Examine the Effect of Different Parameters (No Implemention Needed)**  
After finishing the last two parts, run ``./decision_tree.sh``, which will generate a few plots to examine the effect of the depth of the tree models as well as the size of ensemble. Use these plots to get an idea of how these parameters affect the performance (and whehter your implementation seems correct). Note: 

- This might take up to 15 miniutes to finish.
- If you are running this on Vocareum directly, you need to refresh the page for the png files to show up (so running it locally on your computer is recommended).

