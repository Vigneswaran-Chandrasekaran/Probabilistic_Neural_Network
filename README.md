# Probablistic-Neural-Network
PNN simple code for finding accuracy and Confusion Matrix. 

This simple code for PNN gets two inputs from the user.
1)sample_class_count=gets number of training samples in each class.
2)no_of_class=gets total number of classes in your model.

The input file is of .xlsx form (or) it can be used in other according to need (for .csv use pandas dataframe).

First row of the dataset can contain the feature names for easier working with the dataset and for feature selection.

The input dataset file must have testing vectors after (sample_class_count*no_of_class) ie first row of test sample must start from 451 for (90-10) 5-multiclass 
classfication.

No sigma parameter is implemented, if its needed it can easily be added direct. Since the code reflects direct implementaion of PNN formula.


