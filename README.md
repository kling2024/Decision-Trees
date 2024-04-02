# Decision Trees

Here, I implemented a decision tree from scratch and apply it to the task of classifying whether a mushroom is edible or poisonous.

## Packages 

First I imported all the packages that I needed, including numpy (the fundamental package for scientific computing with Python), matplotlib (a popular library to plot graphs in Python) and``utils.py`` contains helper functions for this assignment.

```python
import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *
%matplotlib inline
```

## Problem Statement

Suppose I am starting a company that grows and sells wild mushrooms. 
- Since not all mushrooms are edible, I would like to be able to tell whether a given mushroom is edible or poisonous based on it's physical attributes
- I have some existing data that we can use for this task. 

## Dataset

I started by loading the dataset for this task. The dataset I have collected is as follows:

| Cap Color | Stalk Shape | Solitary | Edible |
|:---------:|:-----------:|:--------:|:------:|
|   Brown   |   Tapering  |    Yes   |    1   |
|   Brown   |  Enlarging  |    Yes   |    1   |
|   Brown   |  Enlarging  |    No    |    0   |
|   Brown   |  Enlarging  |    No    |    0   |
|   Brown   |   Tapering  |    Yes   |    1   |
|    Red    |   Tapering  |    Yes   |    0   |
|    Red    |  Enlarging  |    No    |    0   |
|   Brown   |  Enlarging  |    Yes   |    1   |
|    Red    |   Tapering  |    No    |    1   |
|   Brown   |  Enlarging  |    No    |    0   |


I have 10 examples of mushrooms. For each example, I have three features: 1) Cap Color (`Brown` or `Red`); 2) Stalk Shape (`Tapering (as in \/)` or `Enlarging (as in /\)`); 3) Solitary (`Yes` or `No`). Label: Edible (`1` indicating yes or `0` indicating poisonous)

For ease of implementation, I have one-hot encoded the features (turned them into 0 or 1 valued features)

| Brown Cap | Tapering Stalk Shape | Solitary | Edible |
|:---------:|:--------------------:|:--------:|:------:|
|     1     |           1          |     1    |    1   |
|     1     |           0          |     1    |    1   |
|     1     |           0          |     0    |    0   |
|     1     |           0          |     0    |    0   |
|     1     |           1          |     1    |    1   |
|     0     |           1          |     1    |    0   |
|     0     |           0          |     0    |    0   |
|     1     |           0          |     1    |    1   |
|     0     |           1          |     0    |    1   |
|     1     |           0          |     0    |    0   |


Therefore, `X_train` contains three features for each example: 1) Brown Color (A value of `1` indicates "Brown" cap color and `0` indicates "Red" cap color); 2) Tapering Shape (A value of `1` indicates "Tapering Stalk Shape" and `0` indicates "Enlarging" stalk shape); 3)Solitary  (A value of `1` indicates "Yes" and `0` indicates "No"). `y_train` is whether the mushroom is edible: 1)`y = 1` indicates edible; 2) `y = 0` indicates poisonous.

```python
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
```
The shape of Xtrain is: (10, 3), the shape of ytrain is: (10,), and number of training examples (m): 10


## Decision Tree

I built a decision tree based on the dataset provided.

the steps for building a decision tree are as follows: 1) Start with all examples at the root node; 2) Calculate information gain for splitting on all possible features, and pick the one with the highest information gain; 3) Split dataset according to the selected feature, and create left and right branches of the tree; 4) Keep repeating splitting process until stopping criteria is met.
  
I implemented the following functions, which letted me split a node into left and right branches using the feature with the highest information gain: 1) Calculate the entropy at a node; 2) Split the dataset at a node into left and right branches based on a given feature; 3) Calculate the information gain from splitting on a given feature 4) Choose the feature that maximizes information gain
    
I used the helper functions I had implemented to build a decision tree by repeating the splitting process until the stopping criteria is met. For this lab, the stopping criteria I have chosen is setting a maximum depth of 2

### Calculate entropy

First, I writed a helper function called `compute_entropy` that computes the entropy (measure of impurity) at a node. The function takes in a numpy array (`y`) that indicates whether the examples in that node are edible (`1`) or poisonous(`0`).

I Completed the `compute_entropy()` function below to: 1) Compute $p_1$, which is the fraction of examples that are edible (i.e. have value = `1` in `y`); 2) The entropy is then calculated as:

$$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$

Note: 1) The log is calculated with base $2$; 2) For implementation purposes, $0\text{log}_2(0) = 0$. That is, if `p_1 = 0` or `p_1 = 1`, set the entropy to `0`; 3) Make sure to check that the data at a node is not empty (i.e. `len(y) != 0`). Return `0` if it is

I completed the `compute_entropy()` function using the previous instructions.

```python
# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    if len(y) != 0:
        p_1 = len(y[y == 1]) / len(y) 
        if p_1 != 0 and p_1 != 1:
             entropy = -p_1 * np.log2(p_1) - (1 - p_1) * np.log2(1 - p_1)
        else:
             entropy = 0
    ### END CODE HERE ###        
    
    return entropy
```

Then I checked if my implementation was correct by running the following test code:

```python
# Compute entropy at the root node (i.e. with all examples)
# Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"

print("Entropy at root node: ", compute_entropy(y_train)) 

# UNIT TESTS
compute_entropy_test(compute_entropy)
```

    Entropy at root node:  1.0
    All tests passed. 
    
Here, the result is All tests passed.

### Split dataset

Next, I wrote a helper function called `split_dataset` that takes in the data at a node and a feature to split on and splits it into left and right branches. I implemented code to calculate how good the split is. 1) The function takes in the training data, the list of indices of data points at that node, along with the feature to split on. 2) It splits the data and returns the subset of indices at the left and the right branch.
For example, I started at the root node (so `node_indices = [0,1,2,3,4,5,6,7,8,9]`), and I chose to split on feature `0`, which is whether or not the example has a brown cap. The output of the function is then, `left_indices = [0,1,2,3,4,7,9]` (data points with brown cap) and `right_indices = [5,6,8]` (data points without a brown cap)
    
    
|           | Brown Cap | Tapering Stalk Shape | Solitary | Edible |
|:---------:|:---------:|:--------------------:|:--------:|:------:|
|     0     |     1     |           1          |     1    |    1   |
|     1     |     1     |           0          |     1    |    1   |
|     2     |     1     |           0          |     0    |    0   |
|     3     |     1     |           0          |     0    |    0   |
|     4     |     1     |           1          |     1    |    1   |
|     5     |     0     |           1          |     1    |    0   |
|     6     |     0     |           0          |     0    |    0   |
|     7     |     1     |           0          |     1    |    1   |
|     8     |     0     |           1          |     0    |    1   |
|     9     |     1     |           0          |     0    |    0   |
    

I completed the `split_dataset()` function shown below. For each index in `node_indices`: 1) If the value of `X` at that index for that feature is `1`, add the index to `left_indices`; 2) If the value of `X` at that index for that feature is `0`, add the index to `right_indices`.

```python
# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    left_indices = []
    right_indices = []
    
    ### START CODE HERE ###
    for i in node_indices:   
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)          
    ### END CODE HERE ###
        
    return left_indices, right_indices
```

Then I checked my implementation using the code blocks below. I tried splitting the dataset at the root node, which contains all examples at feature 0 (Brown Cap). I had provided a helper function to visualize the output of the split.


```python
# Case 1

root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Feel free to play around with these variables
# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("CASE 1:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
# generate_split_viz(root_indices, left_indices, right_indices, feature)

print()

# Case 2

root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)

print("CASE 2:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
# generate_split_viz(root_indices_subset, left_indices, right_indices, feature)

# UNIT TESTS    
split_dataset_test(split_dataset)
```
```
    CASE 1:
    Left indices:  [0, 1, 2, 3, 4, 7, 9]
    Right indices:  [5, 6, 8]
    
    CASE 2:
    Left indices:  [0, 2, 4]
    Right indices:  [6, 8]
    All tests passed.
```

Here, the result is All tests passed.



### Calculate information gain

Next, I wrote a function called `information_gain` that took in the training data, the indices at a node and a feature to split on and returned the information gain from the split.

I completed the `compute_information_gain()` function shown below to compute

$$\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$

Here, 1) $H(p_1^\text{node})$ is entropy at the node; 2) $H(p_1^\text{left})$ and $H(p_1^\text{right})$ are the entropies at the left and the right branches resulting from the split; 3) $w^{\text{left}}$ and $w^{\text{right}}$ are the proportion of examples at the left and right branch, respectively.

```python
# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):           Index of feature to split on
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    ### START CODE HERE ###
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
   
    w_left = len(X_left) / len(X_node)
    w_right = len(X_right) / len(X_node)
    
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    
    information_gain = node_entropy - weighted_entropy    
    ### END CODE HERE ###  
    
    return information_gain
```

Then I checked my implementation using the cell below and calculate what the information gain would be from splitting on each of the featues.

```python
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# UNIT TESTS
compute_information_gain_test(compute_information_gain)
```

    Information Gain from splitting the root on brown cap:  0.034851554559677034
    Information Gain from splitting the root on tapering stalk shape:  0.12451124978365313
    Information Gain from splitting the root on solitary:  0.2780719051126377
    All tests passed.

Here, the result is All tests passed. Splitting on "Solitary" (feature = 2) at the root node gives the maximum information gain. Therefore, it's the best feature to split on at the root node.


### Get best split
I wrote a function to get the best feature to split on by computing the information gain from each feature as I did above and returning the feature that gives the maximum information gain

I completed the `get_best_split()` function shown below. 1) The function takes in the training data, along with the indices of datapoint at that node; 2) The output of the function is the feature that gives the maximum information gain; 3) I could use the `compute_information_gain()` function to iterate through the features and calculate the information for each feature.


```python
# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    
    ### START CODE HERE ###
    max_info_gain=0
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature                 
    ### END CODE HERE ##    
   
    return best_feature
```

Then, I checked the implementation of my function using the cell below.

```python
best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

# UNIT TESTS
get_best_split_test(get_best_split)
```

    Best feature to split on: 2
    All tests passed.


Here, the result is All tests passed. As we saw above, the function returns that the best feature to split on at the root node is feature 2 ("Solitary").

## Building the tree

Here, I used the functions I implemented above to generate a decision tree by successively picking the best feature to split on until I reached the stopping criteria (maximum depth is 2).

```python
# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
```


```python
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
# generate_tree_viz(root_indices, y_train, tree)
```

     Depth 0, Root: Split on feature: 2
    - Depth 1, Left: Split on feature: 0
      -- Left leaf node with indices [0, 1, 4, 7]
      -- Right leaf node with indices [5]
    - Depth 1, Right: Split on feature: 1
      -- Left leaf node with indices [8]
      -- Right leaf node with indices [2, 3, 6, 9]
      

