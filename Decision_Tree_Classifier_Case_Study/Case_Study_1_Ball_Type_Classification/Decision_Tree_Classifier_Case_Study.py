
# Decision tree classifier case study
#
# Consider below characteristics of Machine Learning Application :

# Classifier : 			    Decision Tree
# DataSet : 				Balls Dataset
# Features : 				Weight & Surface type
# Labels : 				    Tennis and Cricket
# Training Dataset : 		15 Entries
# Testing Dataset : 		1 Entry

from sklearn import tree

# Rough 1
# Smooth 0

# Tennis 1
# Cricket 2

def SPH_ML(weight, surface):
    #Step1 & 2
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],
        [35,1],[92,0],[35,1],[35,1],[35,1],
        [96,0],[43,1],[110,0],[35,1],[95,0] ]
        
    Labels = [ 1,1,2,1,2,
        1,2,1,1,1,
        2,1,2,1,2]
    
    #Step 3: Decide Algorithm
    dobj = tree.DecisionTreeClassifier()
    
    #Step 4: Training
    dobj = dobj.fit(Features,Labels)
    
    #Step 5: 
    result = dobj.predict([[weight,surface]])
    
    if result == 1:
        print("Ball is Tennis", dobj)
    else:
        print("Ball is cricket", dobj)

def main():
    print("***** Supervised MachineLearning *****")
    weight = int(input("Enter weight of object "))
    surface = input("Enter surface type of object ")
    
    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Invalid Input ")
        return
        
    SPH_ML(weight,surface)
    
    
    
if __name__ == "__main__":
    main()
    
