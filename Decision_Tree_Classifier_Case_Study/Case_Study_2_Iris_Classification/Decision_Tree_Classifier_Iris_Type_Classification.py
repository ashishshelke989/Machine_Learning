# Decision Tree Classifier - Iris classification Case Study 

# Consider below characteristics of Machine Learning Application :

# Classifier : 			    Decision tree Classifier (Supervised Learning)
# DataSet : 				Iris Dataset
# Features : 				sapel length (cm), sapel width (cm), patal length, (cm) Patel width (cm)
# Labels : 				    'setosa', 'versicolor', 'virginica'
# Volume of Dataset :       150 Entries
# Training Dataset : 		147 Entries
# Testing Dataset : 		3 Entry

from sklearn.datasets import load_iris  # Iris dataset is provided by sklearn
import numpy as np
from sklearn import tree

def main():
    dataset = load_iris()
    
    print("Features of datasets")
    print(dataset.feature_names)
    
    print("Target names of datasets")
    print(dataset.target_names)
    
    # print("Isris data set is :")
    
    # for iCnt in range(len(dataset.target)):
        # print("ID: %d Feature: %s Label: %s" %(iCnt, dataset.data[iCnt], dataset.target[iCnt]))
    
    index = [1, 51, 101] # Dataset entries for testing
    
    test_target = dataset.target[index]
    test_feature = dataset.data[index]
   
    train_target = np.delete(dataset.target, index) # Dataset entries for training
    train_feature = np.delete(dataset.data, index, axis = 0) # Dataset entries for training

    obj = tree.DecisionTreeClassifier()
    
    obj.fit(train_feature, train_target)
    
    result = obj.predict(test_feature)
          
    print("Result prediction by ML", result)
    
    print("Expected Result", test_target)
    
if __name__ == "__main__":
    main()