# X-from-scratch
<b>X-from-scratch is a collection of simple implementations of machine learning algorithms and data science models that I am building from scratch to understand their inner workings and gain deeper insight into how they function. Each implementation is written in Python with minimal dependencies, aiming to showcase the fundamentals behind algorithms</b>

\section{Linear Regression}
Code compares approaches to calculate parameters of linear regression. Small dataset with visible linear dependency between variables covers years of experience against salary. Code firstly calculates parameters using Normal equation (formula derived from MLE) also known as Least square method (loss function is sum of squared residuals). If the Gauss-Markov assumtions are met, the Linear regression model is BLUE (best linear unbiased estimate). Code focuses on calculating the parameters and thus doesn't perform any statistical test to ensure all 4 of Gauss-Markov assumtions are met as it is not it's subject.

Code output
![image](https://github.com/user-attachments/assets/a0660b1b-77e3-4bc7-8efa-41605a3e5f6f)
Graph comparing normal eq. fit with Gradient descent after 25 epochs.

![image](https://github.com/user-attachments/assets/8708890f-c2f3-4c4e-b6c4-c312bf379289)
Graph comparing Gradient descent results after more epochs.

Code also outputs loss graph over epochs.
Lastly for comparison:
Residuals Normal eq.: 938128551.67      
Residuals GD (250 epochs): 2219381293.13
Residuals GD (5000 epochs): 938128554.02




