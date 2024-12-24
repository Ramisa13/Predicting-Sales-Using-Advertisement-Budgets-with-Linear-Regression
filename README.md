# Predicting-Sales-Using-Advertisement-Budgets-with-Linear-Regression
#How to load data and create a train/test split 
#How to build your own Pytorch model for simple linear regression problem on the generated data.
#Training the model with gradient descent algorithm in Pytorch. Simple visualisation of data, loss and linear model
## Part 1. Load dataset and split into training and testing sets

Suppose we observe a set of n real-valued input variables x = $\{x_n\}$ and wish to use this observation to predict
the value of a real-valued target variable $y$. In the tutorial we considered artifical examples with synthetic data. In this exercise we will consider a simple dataset that is suitable for linear regression.

We will take a small advertisement dataset. It contains data on the budget allocated for TV, radio and newspaper advertisements with the resulting sales. It contains n = 200 samples with three variables "TV", "Radio","Newspaper" and the value we want to predict, that is "Sales". Assuming a linear model is a good representation of the correlation between advertisement budgets and final sales, we will try to train a linear model to regress the sales based on advertisement budget.

The dataset is located in the advertisement .csv file and opened as a pandas Dataframe. Firstly we will turn the input variables and regression target into Tensors, and then split the data into training and testing sets. In this case we choose a five fold split, meaning we will use 20% of the data for testing (last 40 samples) and 80% for training (first 160 samples).

Please create the required training and testing data below and plot the relationship between the feature 'TV' and the target. Since we have 3 different features, we will only pick 'TV' a 2D plot.

## Part 2. Solving the linear regression problem in Pytorch using Gradient Descent Algorithm
Congratulations you have prepared the data correctly! Now we will move onto creating our model and training it.
### 2.1 Model
In this part, you will define your own model class. To do that, you have to remember the following rules:
1. The model class should be inherited from [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module);
2. Re-write the **\_\_init\_\_** function and the **forward** function;
3. In the **\_\_init\_\_** function, you should always call the parent's **\_\_init\_\_** function first.
4. Don't use the nn.Linear() layer, implement it yourself.
5. Use 1 tensor to define w (not 3) and 1 tensor to define b.
6. Only torch functions and no iterations inside the model.

To make it simpler, since we are using a linear function to approximate the phenomenon that generated the data, our function will be:

\begin{align*}
y = w^T x + b
\end{align*}

Therefore, in the model, we need to set 2 parameters: $w$, $b$

### 2.2 Training
Here, you will train your model based on the training data and evaluate the model on testing data.
1. Set proper number of iterations and learning rate.
2. Remember to use a proper optimizer (you may have many choices: Adam, SGD, RMSprop, ... please find the detailed information in https://pytorch.org/docs/stable/optim.html and know how to use them).
3. In order to train the model, a loss function should be defined:
\begin{align*}
loss = \frac{1}{N}\sum_{i=1}^{N}|f_i - y_i|,
\end{align*}
where, $f_i$ is the output of the model and $N$ is the number of training data pairs.
4. The model must be trained only using training data.
5. Remember to clear the old gradients of parameters before a new backward propagation.
6. In every certain number of iterations, print the values of the parameters, the training loss, and the testing loss.
7. Meanwhile, please track the training loss and the testing loss in each iteration. Once the training is done, the curves of losses should be plotted (two curves are drawn in the same figure, where x axis indicates iterations and y axis indicates the losses).
8. Lastly, draw all the training data, testing data and the curve of the trained model in the same figure (use different showing styles to distinguish them).
9. 
