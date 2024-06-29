import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt

class NearestNeighbor:
    def __init__(self):
        pass

    def loadData(self):
        def displayFeatureStatastics(X):
            _, ax = plt.subplots(2, 3, figsize=(15, 8))
            for i, feature in enumerate(X):
                ax[i//3, i%3].hist(X[feature], bins=len(X[feature].value_counts()))
                ax[i//3, i%3].set_title(feature)

            plt.tight_layout()
            plt.show()

        # read data
        df = pd.read_csv('titanic_data.csv')
        # split features and label
        X = df.drop(['Survived'], axis=1)
        y = df['Survived']
        # display feature statistics
        displayFeatureStatastics(X)

        return X, y

    def normalize(self, X):
        standard_mean = {}
        standard_std = {}
        # normalize data
        for feature in X:
            standard_mean[feature] = X[feature].mean()
            standard_std[feature] = X[feature].std()
            X[feature] = (X[feature] - X[feature].mean()) / X[feature].std()

        return X, pd.Series(standard_mean), pd.Series(standard_std)

    def distance(self, x1, x2):
        dist = 0
        for feature in x1.keys():
            dist += (x1[feature] - x2[feature])**2
        dist = np.sqrt(dist)   

        return dist

    def KNN(self, X, y, new_x, k):
        # calculate distances
        dist_series = X.apply(lambda x: self.distance(x, new_x), axis=1)
        knn_index = dist_series.nsmallest(k).index
        vote = y[knn_index].value_counts()
        return vote

    def predict(self, X, y, new_x):
        # predict
        predictions = []
        vote_ratio = []
        best_k = 0
        max_vote_ratio = 0
        for k in range(1, len(X)):
            vote = self.KNN(X, y, new_x, k)
            ratio = vote.iloc[0] / k
            if ratio >= max_vote_ratio:
                max_vote_ratio = ratio
                best_k = k
            vote_ratio.append(ratio)
            # save prediction
            prediction = vote.keys()[0]
            predictions.append(prediction)

        x = range(1, len(X))
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        # plot predictions
        ax[0].plot(x, predictions)
        ax[0].set_xlabel('K')
        ax[0].set_ylabel('Prediction')
        ax[0].set_title('KNN Predictions for different K')
        # plot vote ratio
        ax[1].plot(x, vote_ratio)
        ax[1].set_xlabel('K')
        ax[1].set_ylabel('Vote Ratio')
        ax[1].set_title('Vote Ratio for different K')

        plt.show()

        return best_k

    def splitKFold(self, X, y, nfolds=10):
        fold_size = len(X) // nfolds
        kfolds = []
        for i in range(nfolds):
            start = i * fold_size
            end = len(X) if i == nfolds-1 else (i + 1) * fold_size
            X_test = X.iloc[start:end]
            y_test = y.iloc[start:end]
            X_train = X.drop(X.index[start:end])
            y_train = y.drop(y.index[start:end])
            kfolds.append((X_train, y_train, X_test, y_test))

        return kfolds
        
    def crossValidation(self, kfolds, k):
        nfolds = len(kfolds)
        accuracy, precision, recall = 0, 0, 0
        for i in range(nfolds):
            X_train, y_train, X_test, y_test = kfolds[i]
            # test
            tp, fp, tn, fn = 0, 0, 0, 0
            for j in range(len(X_test)):
                vote = self.KNN(X_train, y_train, X_test.iloc[j], k)
                prediction = vote.keys()[0]
                if prediction == 1 and y_test.iloc[j] == 1:
                    tp += 1
                elif prediction == 1 and y_test.iloc[j] == 0:
                    fp += 1
                elif prediction == 0 and y_test.iloc[j] == 1:
                    fn += 1
                else:
                    tn += 1
            accuracy += (tp + tn) / len(X_test)
            precision += tp / (tp + fp) if tp + fp > 0 else 0
            recall += tp / (tp + fn) if tp + fn > 0 else 0
        
        avg_accuracy = accuracy / nfolds
        avg_precision = precision / nfolds
        avg_recall = recall / nfolds

        return avg_accuracy, avg_precision, avg_recall
          

class NaiveBayes:
    def __init__(self):
        self.bernoulli_and_multinomial_features = ['Pclass', 'Sex']
        self.poisson_features = ['Siblings/Spouses Aboard', 'Parents/Children Aboard']
        self.exponential_features = ['Fare']
        self.gaussian_features = ['Age']

    def loadData(self):
        # read data
        df = pd.read_csv('titanic_data.csv')
        # split features and label
        X = df.drop(['Survived'], axis=1)
        y = df['Survived']

        return X, y
       
    def mle_prob(self, X, y, new_x, feature, label):
        # poisson probability density function
        def poisson(x, mean):
            return (mean**x * np.exp(-mean)) / math.factorial(int(x))
        
        # exponential probability density function
        def exponential(x, mean):
            return (1 / mean) * np.exp(-x / mean)
        
        # guassian probability density function
        def guassian(x, mean, std):
            return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
        
        cond_prob = 0
        index = y[y == label].index
        # calculate conditional probability
        if feature in self.poisson_features:
            mean = X.loc[index][feature].mean()
            cond_prob = poisson(new_x[feature], mean)
        elif feature in self.exponential_features:
            mean = X.loc[index][feature].mean()
            cond_prob = exponential(new_x[feature], mean)
        elif feature in self.gaussian_features:
            mean = X.loc[index][feature].mean()
            std = X.loc[index][feature].std()
            cond_prob = guassian(new_x[feature], mean, std)
        elif feature in self.bernoulli_and_multinomial_features:
            # laplace smoothing
            cond_prob += 1
            for i in range(len(X)):
                if y.iloc[i] == label and X.iloc[i][feature] == new_x[feature]:
                    cond_prob += 1
            cond_prob /= (y.value_counts()[label] + X[feature].nunique())

        return cond_prob
    
    
    def predict(self, X, y, new_x):
        # predict
        post_prob_dict = {}

        # calculate prior
        prior_prob_series = y.value_counts() / len(y)
        for label in prior_prob_series.keys():
            prior_prob = prior_prob_series[label]
            # calculate conditional probability
            cond_prob = 1
            for feature in X:
                cond_prob *= self.mle_prob(X, y, new_x, feature, label)
            
            # calculate posterior probability
            post_prob = prior_prob * cond_prob

            post_prob_dict[label] = post_prob

        prediciton = max(post_prob_dict, key=post_prob_dict.get)
        return prediciton, post_prob_dict
    
    def splitKFold(self, X, y, k=10):
        fold_size = len(X) // k
        kfolds = []
        for i in range(k):
            start = i * fold_size
            end = len(X) if i == k-1 else (i + 1) * fold_size
            X_test = X.iloc[start:end]
            y_test = y.iloc[start:end]
            X_train = X.drop(X.index[start:end])
            y_train = y.drop(y.index[start:end])
            kfolds.append((X_train, y_train, X_test, y_test))

        return kfolds
        
    def crossValidation(self, kfolds):
        k = len(kfolds)
        accuracy, precision, recall = 0, 0, 0
        for i in range(k):
            X_train, y_train, X_test, y_test = kfolds[i]
            # test
            tp, fp, tn, fn = 0, 0, 0, 0
            for j in range(len(X_test)):
                prediction, _ = self.predict(X_train, y_train, X_test.iloc[j])
                if prediction == 1 and y_test.iloc[j] == 1:
                    tp += 1
                elif prediction == 1 and y_test.iloc[j] == 0:
                    fp += 1
                elif prediction == 0 and y_test.iloc[j] == 1:
                    fn += 1
                else:
                    tn += 1
            accuracy += (tp + tn) / len(X_test)
            precision += tp / (tp + fp) if tp + fp > 0 else 0
            recall += tp / (tp + fn) if tp + fn > 0 else 0
            print(f'Fold {i}: accuracy = {(tp + tn) / len(X_test)}, precision = {tp / (tp + fp) if tp + fp > 0 else 0}, recall = {tp / (tp + fn) if tp + fn > 0 else 0}')
        
        avg_accuracy = accuracy / k
        avg_precision = precision / k
        avg_recall = recall / k

        return avg_accuracy, avg_precision, avg_recall  


if __name__ == '__main__':
    # Nearest Neighbor
    NN = NearestNeighbor()
    X, y = NN.loadData()
    print(X.head())
    # normalize dataset
    X, standard_mean, standard_std = NN.normalize(X)
    
    # predict my feature vector
    new_x = pd.Series({'Pclass': 3, 'Sex': 0, 'Age': 23, 'Siblings/Spouses Aboard': 0, 'Parents/Children Aboard': 0, 'Fare': 7.75})
    # normalize new_x
    for feature in X:
        new_x[feature] = (new_x[feature] - standard_mean[feature]) / standard_std[feature]
    
    # predict
    best_k = NN.predict(X, y, new_x)
    print(f'Best K = {best_k}')

    # assess the performance of KNN
    avg_accuracy, avg_precision, avg_recall = NN.crossValidation(NN.splitKFold(X, y), best_k)
    print(f"Cross Validation for KNN with k={best_k}")
    print(f'avg_accuracy = {avg_accuracy}, avg_precision = {avg_precision}, avg_recall = {avg_recall}')

    # ----------------------------------------------
    print('----------------------------------------------')
    # Naive Bayes
    NB = NaiveBayes()
    X, y = NB.loadData()

    # predict my feature vector
    new_x = pd.Series({'Pclass': 3, 'Sex': 0, 'Age': 23, 'Siblings/Spouses Aboard': 0, 'Parents/Children Aboard': 0, 'Fare': 7.75})
    prediction, post_prob_dict = NB.predict(X, y, new_x)
    result = 'Survived' if prediction == 1 else 'Deceased'
    print(f'post_prob_dict = {post_prob_dict}, prediction = {prediction}->{result}')

    # cross validation
    avg_accuracy, avg_precision, avg_recall = NB.crossValidation(NB.splitKFold(X, y))
    print("Cross Validation for Naive Bayes")
    print(f'avg_accuracy = {avg_accuracy}, avg_precision = {avg_precision}, avg_recall = {avg_recall}')

