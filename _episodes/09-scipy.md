---
title: Scipy and statsmodels
teaching: 30
exercises: 0
objectivs:
- Define a function that takes parameters.
- Return a value from a function.
- Test and debug a function.
- Set default values for function parameters.
- Explain why we should divide programs into small, single-purpose functions.

questions:
- How can I define new functions?
- What's the difference between defining and calling a function?
- What happens when I call a function?

---


## Regression with Scipy


Regression analysis is a cornerstone of data science, providing a powerful way to model and analyze relationships between variables and forecast outcomes. It operates on the principle of finding the best-fitting line or curve that represents the trend of data points within a dataset. This modeling allows data scientists to not only understand which variables are significant predictors but also to quantify the strength of the impact these predictors have on the response variable. It is an indispensable tool in predictive analytics, enabling the construction of models that can forecast trends, determine the strength of predictors, and even drive decision-making processes by providing concrete, actionable insights.

The linear least-squares fitting method is pivotal in data analysis, offering a robust approach for determining the linear equation that most closely approximates the relationship between two variables. This technique minimizes the sum of the squares of the vertical deviations from each data point to the line, hence the name 'least squares.' By focusing on the smallest aggregate distance, it yields the line of best fit, which serves as a predictive model and a tool for inference. The elegance of the least-squares fit lies in its simplicity and efficiency, which allows for straightforward interpretation and computation. It provides a foundation for understanding more complex relationships by starting with a linear approximation. In practice, the linear least-squares method not only aids in prediction but also in evaluating the strength of associations, helping to discern underlying patterns in data across scientific, industrial, and research disciplines. Its central role is further cemented by its inclusion in fundamental statistical learning techniques, where it acts as a stepping stone to more advanced methods.

`scipy.stats.linregress` is a function from the SciPy library that performs a linear least-squares regression for two sets of measurements.

Here's a step-by-step explanation of how to use `scipy.stats.linregress` for a linear fit:

Import the function: You start by importing `linregress` from `scipy.stats`.
Prepare your data: You should have two arrays of data, `x` and `y`, where `x` is the independent variable and `y` is the dependent variable you're trying to predict.
Execute the regression: Call `linregress` with your `x` and `y` data arrays. The function will return a `LinregressResult` object that contains the results of the regression.
Extract results: From the result object, you can extract the slope, intercept,  𝑟 -value (coefficient of determination),  𝑝 -value (hypothesis test for non-correlation), and standard error of the estimated gradient.
Here's an example using `scipy.stats.linregress`:

~~~
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])  # Hours studied
y = np.array([2, 3, 5, 7, 11]) # Exam score

# Perform linear regression
fit1 = linregress(x, y)
slope = fit1.slope 
intercept = fit1.intercept 
# Use the slope and intercept to construct the best fit line
y_pred = fit1.intercept + fit1.slope * x

# Plotting the data points
plt.scatter(x, y, color='blue', label='Data points')

# Plotting the best fit line
plt.plot(x, y_pred, color='red', label=f'Best fit line: y={slope:.2f}x+{intercept:.2f}')

# Adding the legend
plt.legend()
~~~
{: .python}


Scikit-learn, often abbreviated as sklearn, is a versatile and widely-used open-source machine learning library for Python. It provides a range of supervised and unsupervised learning algorithms through a consistent interface. Essential tools for data mining and data analysis are also part of this library, making it a go-to choice for many data scientists and researchers. Scikit-learn is built upon the SciPy (Scientific Python) ecosystem and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. Its simplicity, accessibility, and broad range of algorithms make it very useful for educational purposes, yet it's powerful enough to be used in large-scale industrial applications. Scikit-learn includes a variety of algorithms for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing. Its robustness, ease of use, and good documentation have made it one of the most popular machine learning libraries in the world.

~~~
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable (feature), needs to be 2D for sklearn
y = np.array([2, 3, 5, 7, 11])           # Dependent variable (target), 1D is fine

# Create an instance of the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# After fitting the model, we can check the slope (coef_) and intercept (intercept_)
slope = model.coef_[0]
intercept = model.intercept_

# Make predictions (optional)
X_new = np.array([[6], [7]])  # New data for prediction
y_pred = model.predict(X_new)

# slope, intercept, and predictions are now available
print(f"Slope: {slope}, Intercept: {intercept}, Predictions: {y_pred}")
~~~



## Gap analysis with Continuous Variables

Gap Analysis determines whether two samples of data are different. In our running example, we want to determine whether Sample 1 (salaries of female employees in the bank) is different from Sample 2 (salaries of male employees at the bank). We generally come at Gap Analysis in two steps:

Plot the data in such a way that we can visually assess whether a gap exists. These visualizations also come in handy later when communicating the results of any formal analysis.

Conduct a formal gap analysis using statistical techniques.


~~~
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# For statistics. Requires statsmodels 5.0 or more
from statsmodels.formula.api import ols
# Analysis of Variance (ANOVA) on linear models
from statsmodels.stats.anova import anova_lm


path = 'https://raw.githubusercontent.com/mesfind/datasets/master/banking.csv'
bank = pd.read_csv(path)
bank.head()
~~~
{: .python}

~~~
   age          job  marital          education  default housing loan   contact month day_of_week  duration  campaign  pdays  previous     poutcome  emp_var_rate  cons_price_idx  cons_conf_idx  euribor3m  nr_employed  y
0   44  blue-collar  married           basic.4y  unknown     yes   no  cellular   aug         thu       210         1    999         0  nonexistent           1.4          93.444          -36.1      4.963       5228.1  0
1   53   technician  married            unknown       no      no   no  cellular   nov         fri       138         1    999         0  nonexistent          -0.1          93.200          -42.0      4.021       5195.8  0
2   28   management   single  university.degree       no     yes   no  cellular   jun         thu       339         3      6         2      success          -1.7          94.055          -39.8      0.729       4991.6  1
3   39     services  married        high.school       no      no   no  cellular   apr         fri       185         2    999         0  nonexistent          -1.8          93.075          -47.1      1.405       5099.1  0
4   55      retired  married           basic.4y       no     yes   no  cellular   aug         fri       137         1      3         1      success          -2.9          92.201          -31.4      0.869       5076.2  1

~~~

## Visual gap analysis
The boxplot in Seaborn permits both an x and y axis. For the resulting boxplot to make sense, the x variable must be continuous (like salary) and the y variable must be categorical (like gender). This permits a very quick comparison of the distribution of salary by gender.


~~~
sns.boxplot(x=bank['cons_price_idx'], y=bank['marital'], showmeans=True);
plt.show()
~~~
{: .python}



