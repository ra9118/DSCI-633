## K Nearest Neighbor

### Build your own kNN classifier (with continuous input)

#### Implement my_KNN.fit() function in [my_KNN.py](https://github.com/hil-se/fds/blob/master/assignments/assignment4/my_KNN.py)
Inputs:
- X: pd.DataFrame, independent variables, each value is a continuous number of float type
- y: list, np.array or pd.Series, dependent variables, each value is a category of int or str type

#### Implement my_KNN.predict() function in [my_KNN.py](https://github.com/hil-se/fds/blob/master/assignments/assignment4/my_KNN.py)
Input:
- X: pd.DataFrame, independent variables, each value is a continuous number of float type

Output:
- Predicted categories of each input data point. List of str or int.

#### Implement my_KNN.predict_proba() function in [my_KNN.py](https://github.com/hil-se/fds/blob/master/assignments/assignment4/my_KNN.py)
Input:
- X: pd.DataFrame, independent variables, each value is a continuous number of float type

Output:
- Prediction probabilities of each input data point belonging to each categories. pd.DataFrame(list of prob, columns = self.classes_).

Example:
- self.classes_ = {"2", "1"}
- the 5 nearest neighbors for the test data point have labels of {"1":4, "2":1}
- then the prob for that data point is {"1": 4/5, "2": 1/5}
- return probs = pd.DataFrame(list of prob, columns = self.classes_)

### Test my_KNN classifier with [A4.py](https://github.com/hil-se/fds/blob/master/assignments/assignment4/A4.py)

