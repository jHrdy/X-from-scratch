import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

data = pd.read_csv('Sal_Data.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

X_w_const = np.append(np.ones(X.shape), X, axis=1)

y_mat = np.matrix(y).T     
X_mat = np.matrix(X_w_const)

# calculating coefficients Normal equation (loss: MSE)
coeffs = (inv(X_mat.T*X_mat)*(X_mat.T*y_mat)).A

# function for later evaluation (loss: MSE)
def sum_residuals(a,b):
    yhats = [a*x+b for x in X]
    residuals = [(yhats[i]-y[i])**2 for i in range(len(y))]
    return sum(residuals)[0]

def reg(coeffs):
    return [coeffs[1]*X[i]+coeffs[0] for i in range(len(X))]
    
def grad_descent(epochs, alpha):
    a = 0
    b = 0
    m = len(X)
    cnt = 0
    values = []
    while cnt < epochs:
        dJ_da = (1/m)*sum([(a*X[i]+b - y[i])*X[i] for i in range(m)])
        dJ_db = (1/m)*sum([(a*X[i]+b - y[i]) for i in range(m)])
        a = a - alpha*dJ_da                                # absolútne nechápem prečo nefunguje operátor -=
        b = b - alpha*dJ_db
        values.append((a,b))
        cnt += 1
    return a,b, values


a,b, _ = grad_descent(25, 0.01)
coeffs_from_g_d = (b,a)             # variable suitable as an input for reg() 

plt.plot(X,reg(coeffs), color='green', label="Normal eq. fit")
plt.plot(X,reg(coeffs_from_g_d),color='black', label="Gradient descent fit")
plt.scatter(X,y, label='Salary data')
plt.title('Fit: Normal equation & Gradient descent (25 epochs)')
plt.legend()
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.grid(True)
plt.show()

a_250,b_250, _ = grad_descent(250, 0.01)
coeffs_from_g_d_250 = (b_250,a_250)

a_5k,b_5k, vals_5k = grad_descent(5000, 0.01)
coeffs_from_g_d_5k = (b_5k,a_5k)

plt.scatter(X,y, label='Salary data')
plt.plot(X,reg(coeffs_from_g_d_250),color='black', label="GD 250 epochs")
plt.plot(X,reg(coeffs_from_g_d_5k),color='red', label="GD 5000 epochs")
plt.title('Salary Data, fit: Grad descent (250 & 5000 epochs)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.legend()
plt.grid(True)
plt.show()

print('Reziduals Normal eq.:', round(sum_residuals(coeffs[1], coeffs[0]),2))
print('Reziduals GD (250 epochs):', round(sum_residuals(a_250,b_250),2))
print('Reziduals GD (5000 epochs):', round(sum_residuals(a_5k,b_5k),2))

residuals = [sum_residuals(v[0],v[1]) for v in vals_5k]

plt.title('Training first 1000 epochs')
plt.xlabel('Epochs')    
plt.ylabel('Loss')  
plt.grid(True)  
plt.plot(residuals[:1000])
plt.show()
