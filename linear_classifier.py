import numpy as np
import matplotlib.pyplot as plt


class LinearClassifier:
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    def fit(self,X,y,n_iterations):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(n_iterations):
            linear_output = np.dot(X,self.weights) + self.bias
            p = 1/(1+np.exp(-linear_output))
            dw = 1/n_samples*np.dot(X.T,(p-y))
            db = 1/n_samples*np.sum(p-y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self,X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet.")
        linear_output = np.dot(X,self.weights) + self.bias
        p = 1/(1+ np.exp(-linear_output))
        predicted_classes = [1 if i > 0.5 else 0 for i in p]
        return np.array(predicted_classes)
    def plot_decision_boundary(self,X,y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = self.predict(np.array([xx.ravel(), yy.ravel()]).T)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Linear Classifier Decision Boundary')
        plt.show()
    def accuracy(self,y_true,y_pred):
        return np.sum(y_true == y_pred)/len(y_true)
    def precision(self,y_true,y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        if predicted_positives == 0:
            return 0
        return true_positives / predicted_positives
    def recall(self,y_true,y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        if actual_positives == 0:
            return 0
        return true_positives / actual_positives
    def f1_score(self,y_true,y_pred):
        precision = self.precision(y_true,y_pred)
        recall = self.recall(y_true,y_pred)
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)
        
if __name__ == "__main__":
    from sklearn.datasets import make_moons
    X,y = make_moons(n_samples = 1000,noise = 0.1,random_state = 42)
    classifier = LinearClassifier(learning_rate=0.01)
    classifier.fit(X,y,n_iterations=1000)
    y_pred = classifier.predict(X)
    print("Accuracy:", classifier.accuracy(y,y_pred))
    print("Precision:", classifier.precision(y,y_pred))
    print("Recall:", classifier.recall(y,y_pred))
    print("F1 Score:", classifier.f1_score(y,y_pred))
    classifier.plot_decision_boundary(X,y)  