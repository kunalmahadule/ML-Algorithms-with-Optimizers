# 1 April 2025 GD, SGD, Batch GD, polynomial


import numpy as np
import pandas as pd # ctrl+i--> for help
import matplotlib.pyplot as plt

# dataset = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\SPYDER-ML\Regression\#4  GD,SGD,Batch\emp_sal.csv")
dataset = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\SPYDER-ML\Regression\#4  GD,SGD,Batch+poly+svr,knn+dt,rf\emp_sal.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values



# ***LINEAR REGRESSION MODEL*** 
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Graph for Linear Regression Model
''' 
(the degree of lin_reg is 1/linear means degree is 1)

we not get best fit line. the residual is high. the actual and 
predicted datapoint are not same
'''
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color= 'blue')
plt.title('linear regression model (Linear Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


# slope
m = lin_reg.coef_
print(m)

# intercept
c = lin_reg.intercept_
print(c)


# Model prediction
'''
For 6.5 yrs Experience if the model predict the salary between
150000 - 200000 so my model is good.

but model predicted - 330378.787 which is wrong

'''
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred




# ***POLYNOMIAL REGRESSION MODEL***
# this is non-linear model (Bydefault degree is 2)
 
from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=2)
poly_reg = PolynomialFeatures(degree=5) # for degree 5,6 it works well so freeze it
# poly_reg = PolynomialFeatures(degree=2)



# y = m1x + m2x² + m3x³ + c --> now the x is created into 2 more column with square and cube
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)


# here the linear model with degree is 2 above linear model we build with degree 1
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


# Graph for Polynomial model
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('polymodel (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Model prediction
'''
For 6.5 yrs Experience if the model predict the salary between
150000 - 200000 so my model is good.

but model predicted - 189498.106 so the model is good model
when degree is 2
'''

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred




'''
For emp salary dataset we are going to build all models

linear
poly
svr
knn
dtr
rfr


jo model gives high accuracy us model ko dump or going to deploy using pickle.

In real time also regression, classification for one dataset we build several model
and whatever gives high accuracy with that modle we are going to deployment.

'''




# ============================================================================
'''
Task: My Task is that is to make excel sheet for all these model and 
     record it's result.
     Also make another SVR excel sheet and change these parameters and check its 
     accuracy and store in the sheet.

Model Rank:
    lin reg - 330
    poly reg - 174.8 (175)
    svr_reg - 175.7 (176)
    
    
'''

# 2 April

# *** SUPPORT VECTOR REGRESSION (SVR) MODEL ***

from sklearn.svm import SVR

'''
SVR parameters:
    
kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable, default='rbf'

degreeint, default=3

gamma{'scale', 'auto'} or float, default='scale'

Cfloat, default=1.0 - Regularization parameter
'''
# svr_reg = SVR() # parameter tuning
svr_reg = SVR(kernel='poly',degree=4,gamma='auto',C =10.0) # hyperparameter tuning
svr_reg.fit(x, y)

svr_model_pred = svr_reg.predict([[6.5]])
svr_model_pred






# *** K-NEAREST NEIGHBORS (KNN) MODEL ***
'''
Task: Make Excel sheet for KNN model with all parameters
'''
from sklearn.neighbors import KNeighborsRegressor
'''
parameters of KNeighborsRegressor: 
    
n_neighbors: Bydefault it will ask to 5 neighbors

weights: suppose we have 3 record so the weight of 1 record is 1/3
        this weights is uniform for all 3 records
        
algorithm{'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'

p = 1- for manhattend distance
    2- for euclidian distance

'''
# knn_reg = KNeighborsRegressor() # parameter tuning
knn_reg = KNeighborsRegressor(n_neighbors=7,weights='uniform',algorithm='ball_tree')
knn_reg.fit(x,y)

knn_model_pred  = knn_reg.predict([[6.5]])
knn_model_pred







'''
In future in company always build all ml algo for one dataset


linear regression
polynomial regression
svr regression
knn regression
decision tree
random forest


whatever model give good accuracy so with that go for deployment

'''



# Note: In this dataset we have'nt done splitting because the 
# data is less. if the dataset is less we can build like this way.





# ============================================================================

# 3 April

# *** DECISION TREE REGRESSOR MODEL ***

from sklearn.tree import DecisionTreeRegressor
'''
criterion="squared_error" -> Criteria to build ml model

splitter="best" -> 
'''

# dt_reg = DecisionTreeRegressor() #parameter tuning
dt_reg = DecisionTreeRegressor(criterion='absolute_error', splitter='random') #hyperparamter tuning

dt_reg.fit(x,y)

dt_model_pred = dt_reg.predict([[6.5]])
dt_model_pred


# decision tree & random forest both same





# *** RANDOM FOREST REGRESSOR MODEL ***

from sklearn.ensemble import RandomForestRegressor
'''
n_estimators=100: By default it gives 100 trees to predict


'''
# with this we get different different result
# rf_reg = RandomForestRegressor() #parameter tuning

# note that while working with randomforest
rf_reg = RandomForestRegressor(random_state=0)

# rf_reg = RandomForestRegressor() #Hyperparameter tuning

rf_reg.fit(x,y)


rf_model_pred = rf_reg.predict([[6.5]])
rf_model_pred

    

















'''
Above these algorithms having some numbers of parameters but in LLM there are
billions of parameters are there.

All above model we build is called as Trained model. 
Pre-trained model done by GPT, llm etc.

'''


































































