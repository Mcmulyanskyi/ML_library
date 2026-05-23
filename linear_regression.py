import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self,n_repeats,momentum,regularization,learning_rate= 0.01):
        self.learning_rate = 0.01
        self.n_repeats = n_repeats
        self.momentum = momentum
        self.regularization = regularization
        self.weights = None
    def fit(self,X,y,adaptive_learing_rate = False):
        if adaptive_learing_rate:
            self.calculate_learning_rate(X)
        prev_dw = np.zeros(X.shape[1])
        for i in range(self.n_repeats):
            y_pred = self.predict(X)
            error = y_pred - y
            dw = 1/len(X)*X.T.dot(error) + self.regularization*self.weights
            dw = self.momentum*prev_dw + (1-self.momentum)*dw
            prev_dw = dw
            self.weights = self.weights - self.learning_rate*dw
    def predict(self,X):
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])
        return X.dot(self.weights)
    def calculate_loss(self,X,y):
        if self.weights is not None:
            y_pred = self.predict(X)
            error = y_pred - y
            loss = np.mean(error**2) + self.regularization*np.sum(self.weights**2)
            return loss
        else:
            return float('inf')
    def calculate_learning_rate(self,X):
        learning_rate = np.linalg.eigvals(X.T.dot(X))
        self.learning_rate = 1 / np.max(learning_rate)
if __name__=="__main__":
    from sklearn.datasets import make_regression
    X,y = make_regression( n_samples=100, n_features=1, noise=10)
    best_alpha = None
    best_loss = float('inf')
    for alpha in [0,0.05,0.1,0.15,0.2]:
        model = LinearRegression(n_repeats=1000,momentum=0.9,regularization=alpha)
        model.fit(X,y,adaptive_learing_rate=True)
        loss = model.calculate_loss(X,y)
        if loss < best_loss:
            best_alpha = alpha
            best_loss = loss
        print(f"Regularization: {alpha}, Loss: {loss}")
    print(f"Best Regularization: {best_alpha}, Loss: {loss}")
    plt.plot(X,y,label = 'Data')
    plt.plot(X,model.predict(X),label = 'Predicted')    
    plt.legend()
    plt.show()