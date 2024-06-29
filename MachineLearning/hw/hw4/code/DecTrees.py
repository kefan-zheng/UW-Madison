import numpy as np
import pandas as pd
from graphviz import Digraph
import matplotlib.pyplot as plt

def displayFeatureStatastics(X):
    _, ax = plt.subplots(2, 3, figsize=(15, 8))

    features = X.columns.tolist()
    for i, feature in enumerate(features):
        ax[i//3, i%3].hist(X[feature], bins=len(X[feature].value_counts()))
        ax[i//3, i%3].set_title(feature)
                    
    plt.tight_layout()
    plt.show()

def loadDataSet():
    # read data from file
    df = pd.read_csv('titanic_data.csv')
    # split features and label
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']

    # display data statistics
    displayFeatureStatastics(X)

    # transform each of your features into a binary variable
    # passenger class (one-hot)
    one_hot_column = pd.get_dummies(X['Pclass'], prefix='Pclass').astype('int64')
    X = one_hot_column.join(X)
    X = X.drop(['Pclass'], axis=1)
    # age
    median_age = X['Age'].median()
    X['Age'] = X['Age'].apply(lambda x: 0 if x < median_age else 1)
    # siblings/spouses
    X['Siblings/Spouses Aboard'] = X['Siblings/Spouses Aboard'].apply(lambda x: 0 if x == 0 else 1)
    # parents/children
    X['Parents/Children Aboard'] = X['Parents/Children Aboard'].apply(lambda x: 0 if x == 0 else 1)
    # fare
    median_fare = X['Fare'].median()
    X['Fare'] = X['Fare'].apply(lambda x: 0 if x < median_fare else 1)
    
    return X, y

# split data into k folds
def splitKFold(X, y, k):
    fold_size = X.shape[0] // k
    kfolds = []
    for i in range(k):
        start = i * fold_size
        end = X.shape[0] if i == k-1 else (i + 1) * fold_size
        X_test = X[start:end]
        y_test = y[start:end]
        X_train = X.drop(X.index[start:end])
        y_train = y.drop(y.index[start:end])
        kfolds.append((X_train, y_train, X_test, y_test))

    return kfolds

class Node:
    def __init__(self):
        # split feature
        self.feature = None
        # children
        self.left = None # 0
        self.right = None # 1
        # leaf node
        self.is_leaf = False

class DecisionTree:
    def __init__(self):
        self.root = None

    def entropy(self, x):
        n = x.shape[0]
        entropy = 0
        for k in x.value_counts().keys():
            p_k = x.value_counts()[k] / n
            entropy += p_k * np.log2(1 / p_k)
        return entropy

    # calculate mutual information between x and y
    def mutualInfo(self, x, y):
        # calculate entropy of x
        entropy_x = self.entropy(x)

        # calculate entropy of x|y
        n = x.shape[0]
        entropy_x_y = 0
        for k in y.value_counts().keys():
            num_y_k = y.value_counts()[k]
            p_y_k = num_y_k / n
            x_y_k = x[y == k]
            entropy_x_y_k = 0
            for j in x_y_k.value_counts().keys():
                num_x_j_y_k = x_y_k.value_counts()[j]
                p_x_j_y_k = num_x_j_y_k / num_y_k
                entropy_x_y_k += p_x_j_y_k * np.log2(1 / p_x_j_y_k)
            entropy_x_y += p_y_k * entropy_x_y_k

        # calculate mutual information
        mutual_info = entropy_x - entropy_x_y
        return mutual_info

    # choose the feature with the highest mutual information
    def getBestFeature(self, X, y):
        max_mutual_info = 0
        best_feature = None
        # look each feature
        for feature in X:
            mutual_info = self.mutualInfo(X[feature], y)
            if mutual_info > max_mutual_info:
                max_mutual_info = mutual_info
                best_feature = feature
        
        return best_feature
    
    def splitData(self, X, y, best_feature):
        left_index = (X[best_feature] == 0)
        right_index = (X[best_feature] == 1)
        X = X.drop([best_feature], axis=1)
        # left subset
        left_X = X[left_index]
        left_y = y[left_index]
        # right subset
        right_X = X[right_index]
        right_y = y[right_index]
        return left_X, left_y, right_X, right_y
    
    def ifFinish(self, X, y):
        # check if it is a leaf node
        if y.shape[0] < 44 or (self.entropy(y) <= 0.01) or all(self.mutualInfo(X[feature], y) <= 0.01 for feature in X):
            return True
        return False

    def construct(self, X, y):

        root = Node()

        # check if it is a leaf node
        if self.ifFinish(X, y):
            root.is_leaf = True
            root.feature = y.value_counts(sort=True).keys()[0]
            return root
        
        # choose the feature with the highest mutual information
        best_feature = self.getBestFeature(X, y)
    
        # split the data into two parts
        left_X, left_y, right_X, right_y = self.splitData(X, y, best_feature)
        root.feature = best_feature
        root.left = self.construct(left_X, left_y)
        root.right = self.construct(right_X, right_y)

        self.root = root
        return root
    
    def predict(self, node, x):
        # check if it is a leaf node
        if node.is_leaf:
            return node.feature
        
        feature = node.feature
        node = node.left if x[feature] == 0 else node.right
        return self.predict(node, x)
    
    def visualize(self, node, graph=None):
        if graph is None:
            graph = Digraph()
            graph.node(str(id(node)), str(node.feature))
    
        if node.left:
            graph.node(str(id(node.left)), str(node.left.feature))
            graph.edge(str(id(node)), str(id(node.left)), label='0')
            self.visualize(node.left, graph)
        
        if node.right:
            graph.node(str(id(node.right)), str(node.right.feature))
            graph.edge(str(id(node)), str(id(node.right)), label='1')
            self.visualize(node.right, graph)

        return graph
    
    def crossValidate(self, X, y, k=10):
        # split data into k folds
        kfolds = splitKFold(X, y, k)

        # construct k decision tree
        print("\n10-Fold Cross Validation for decision tree:")
        cross_valid_accuracy = 0
        for i in range(k):
            X_train, y_train, X_test, y_test = kfolds[i]
            tree = DecisionTree()
            tree.construct(X_train, y_train)
            # predict
            y_pred = X_test.apply(lambda x: tree.predict(tree.root, x), axis=1)
            # calculate accuracy
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            print("Fold ", i, " Accuracy :", accuracy)
            # accumulate accuracy
            cross_valid_accuracy += accuracy

        avg_accuracy = cross_valid_accuracy / k

        return avg_accuracy
    
class RandomForest():
    def __init__(self):
        self.trees = []
        self.n_trees = None

    def construct(self, X, y, n_trees=5):
        self.n_trees = n_trees
        trees = []
        for i in range(n_trees):
            # select 80% samples randomly
            samples_index = X.sample(frac=0.8, random_state = i).index
            X_train = X.loc[samples_index]
            y_train = y.loc[samples_index]
            # construct decision tree
            tree = DecisionTree()
            tree.construct(X_train, y_train)
            trees.append(tree)

        self.trees = trees

    def constructX(self, X, y, n_trees=6):
        self.n_trees = n_trees
        trees = []
        for i in range(n_trees):
            # exclude one feature
            if i == 0:
                X_remain = X.drop(X.columns[:i+3], axis=1)
            else:
                X_remain = X.drop(X.columns[i+2], axis=1)
            # select 80% samples randomly
            samples_index = X_remain.sample(frac=0.8, random_state = i).index
            X_train = X_remain.loc[samples_index]
            y_train = y.loc[samples_index]
            # construct decision tree
            tree = DecisionTree()
            tree.construct(X_train, y_train)
            trees.append(tree)

        self.trees = trees

    def predict(self, x):
        y_pred = []
        for tree in self.trees:
            y_pred.append(tree.predict(tree.root, x))
        # majority vote
        random_forest_pred = 1 if sum(y_pred) > self.n_trees // 2 else 0
        return random_forest_pred
        
    def visualize(self):
        graphs = []
        for _, tree in enumerate(self.trees):
            graph = tree.visualize(tree.root)
            graphs.append(graph)
        return graphs
    
    def crossValidate(self, X, y, k=10):
        # split data into k folds
        kfolds = splitKFold(X, y, k)

        # construct k random forest
        print("\n10-Fold Cross Validation for random forest:")
        cross_valid_accuracy = 0
        for i in range(k):
            X_train, y_train, X_test, y_test = kfolds[i]
            randomForest = RandomForest()
            randomForest.construct(X_train, y_train, n_trees=5)
            # predict
            y_pred = X_test.apply(lambda x: randomForest.predict(x), axis=1)
            # calculate accuracy
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            print("Fold ", i, " Accuracy :", accuracy)
            # accumulate accuracy
            cross_valid_accuracy += accuracy

        avg_accuracy = cross_valid_accuracy / k

        return avg_accuracy
    
    def crossValidateX(self, X, y, k=10):
        # split data into k folds
        kfolds = splitKFold(X, y, k)

        # construct k random forest
        print("\n10-Fold Cross Validation for random forest:")
        cross_valid_accuracy = 0
        for i in range(k):
            X_train, y_train, X_test, y_test = kfolds[i]
            randomForest = RandomForest()
            randomForest.constructX(X_train, y_train, n_trees=6)
            # predict
            y_pred = X_test.apply(lambda x: randomForest.predict(x), axis=1)
            # calculate accuracy
            accuracy = np.sum(y_pred == y_test) / len(y_test)
            print("Fold ", i, " Accuracy :", accuracy)
            # accumulate accuracy
            cross_valid_accuracy += accuracy

        avg_accuracy = cross_valid_accuracy / k

        return avg_accuracy


if __name__ == '__main__':
    # load data
    X, y = loadDataSet()
    print("X:\n", X.head())
    print("y:\n", y.head())

    # construct decision tree
    tree = DecisionTree()
    tree.construct(X, y)

    # display decision tree
    graph = tree.visualize(tree.root)
    graph.render('decision_tree', format="png", view=False)

    # cross validation for decision tree
    accuracy = tree.crossValidate(X, y, k=10)
    print("Cross validation accuracy:", accuracy)

    # predict my feature vector
    x = pd.Series({'Pclass_1': 0, 'Pclass_2': 0, 'Pclass_3': 1, 'Sex': 0, 'Age': 0, 'Siblings/Spouses Aboard': 0, 'Parents/Children Aboard': 0, 'Fare': 0})
    y_pred = tree.predict(tree.root, x)
    prediction = "survived" if y_pred == 1 else "deceased"
    print(f"Prediction with decision tree: {prediction}, y_pred: {y_pred}")

    # construct random forest with 5 decison trees
    randomForest = RandomForest()
    randomForest.construct(X, y, n_trees=5)

    # display random forest
    graphs = randomForest.visualize()
    for i, graph in enumerate(graphs):
        graph.render('random_forest_subtree_'+str(i), format="png", view=False)

    # cross validation for random forest
    accuracy = randomForest.crossValidate(X, y, k=10)
    print("Cross validation accuracy:", accuracy)

    # predict my feature vector with random forest
    y_pred = randomForest.predict(x)
    prediction = "survived" if y_pred == 1 else "deceased"
    print(f"Prediction with random forest: {prediction}, y_pred: {y_pred}")

    # construct random forest with 6 decison trees, each excluding one feature
    randomForest = RandomForest()
    randomForest.constructX(X, y, n_trees=6)

    # display random forest with 6 decison trees, each excluding one feature
    graphs = randomForest.visualize()
    for i, graph in enumerate(graphs):
        graph.render('random_forest_subtree_excluding_feature_'+str(i), format="png", view=False)

    # cross validation for random forest with 6 decison trees, each excluding one feature
    accuracy = randomForest.crossValidateX(X, y, k=10)
    print("Cross validation accuracy:", accuracy)

    # predict my feature vector with random forest (excluding feature)
    y_pred = randomForest.predict(x)
    prediction = "survived" if y_pred == 1 else "deceased"
    print(f"Prediction with random forest (excluding feature): {prediction}, y_pred: {y_pred}")
    