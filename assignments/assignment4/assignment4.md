# TDT4171 - Artificial Intelligence Methods
## Assignment 4 - Decision Trees
### Problem 1 - Decision Trees
__a)__ See [assignment4.py](assignment4.py) for the implementation of the decision tree. I decided to drop the columns `Name`, `Ticket` and `Cabin`, since the cabin name of the passenger, the ticket number, and the cabin number do not reflect the likelihood of surviving. These columns are also strings, and may be difficult to categorize as either categorical or continious variables. I also dropped rows where we had missing values. The categorical variables I trained the model on were:
* `Pclass`
* `Sex`
* `SibSp`
* `Parch`
* `Embarked`

One could argue that `SibSp` and `Parch` are continious variables, but these variables only have 4 discrete values, and since I consider `Pclass` as a categorical variable, it is natural that I also consider these variables as categorical.

Thus, the continious variables in the dataset are:
* `Age`
* `Fare`

After fitting my decision tree to the training data, I got a test accuracy of 92.7710843373494 %

Here is a visualization of the decision tree:

![](categorical_decision_tree.png)

__b)__ See [assignment4.py](assignment4.py) for the implementation of the decision tree. After fitting my decision tree to the training data, I got a test accuracy of 93.97590361445783 %

Here is a visualization of the decision tree:

![](decision_tree.png)

__c)__ The performance of the two models is almost the same. This can happen because the predictors used to train the model with only categorical variables may explain most of the variance in the dataset. 

Here are some changes for how we could improve the dataset:
* Extract information from the columns I dropped. For example, the `Name` columns includes the title of each person, for example `Mr.`, `Mrs.`, `Dr.`, `Prof.` and so on. These can reflect the probability of surviving.
* We can also make new features out of the current features. For example, combining `Pclass` and `Age` to see how age and your ticket class together determine your chances of surviving.
* Use PCA (Principal Component Analysis) to remove features which explain little variability in the data.
* Use some of the techniques discussed in __Problem 2__ to fill in missing values. These may have important information to make the model better.
* This may not improve performance, but one can also use pruning to remove redundant or non-important parts of the decision tree to make the model more _interpretable_.

### Problem 2 - Missing Values

__Method I__

One way to handle missing data is to fill the missing values for an attribute. This can be done by for example using regression on the column with missing data, and predict the missing values. Here is some pseudocode for how I would do it

```
X <-- data column with missing values
X_train <-- data column without missing values
regression_model <-- fit(X_train)
X_regression <-- regression_model(X)
X_regression <-- data column where missing values have been predicted
```

The main advantage here is that we might get reasonable values for the missing values, and may get a better decision tree in the end. The disadvantage for this approach is that it won't work for categorical variables. The assumption here is that the values are continious, and that they follow a certain pattern (for example linear). If that was not the case, regression would be difficult to use here.

__Method II__

Another way to handle missing data is to use K-nearest-neighbours to find similar datapoints to the ones where we have a missing value. Then we can set the missing value equal to the value in the neighbourhood. In the Titanic dataset, if two people have almost the same attributes, but one of them is missing `Age`, we could give them the same age. 

The main advantage here is the same as above, that we might get a better model. Another advantage is that this method will also work for categorical variables (we might have to encode those variables so that we can measure _similarity_). The disadvantage here is that we might not get good neighbourhoods. This can lead us to fill in bad values for the missing values, which can give us a poor model. The assumption for this approach is that we assume that our datapoints are similar to other datapoints. Otherwise, the KNN algorithm will set the missing value equal to the nearest datapoint, which in reality can be far away.