import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})

lower = 0
upper = 340

# success
d = pd.read_csv("all_2009-8.csv")
#d = d.sort_values("overs range")
x = d.iloc[ lower : upper , [2 , 4, 5 , 6]].values
y = d.iloc[lower : upper , -1].values



######################### multiple regression#################################

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x, y)

# Predicting the Test set results
#y_pred = regressor.predict(X_test)


#print(regressor.coef_)
#print(regressor.intercept_)

######################### multiple regression#################################




######################### random forest#################################
"""
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 8, random_state = 0)
regressor.fit(x, y)
"""
######################### random forest#################################





lower = 0
upper = 15   ##  changed this value

b = pd.read_csv("2019.csv")### changed this value
x_test = b.iloc[lower : upper  , [1 ,4, 6, 3]].values
y_test = b.iloc[lower : upper , -5].values   
y_pred = regressor.predict(x_test)







################################error compare#########################################
"""
#-3 , -5
import copy
errorC = copy.deepcopy(b.iloc[ : , -5])
errorC2 = copy.deepcopy(b.iloc[ : , -3])

sumC = sum(errorC)
sumC2 = sum(errorC2)

error_C = ((sumC - sumC2)*100)/sumC
print(error_C)


errorO2 = copy.deepcopy(y_pred)
sumO2 = sum(errorO2)

error_O = ((sumC - sumO2)*100)/sumC
print(error_O)





import copy

temp = copy.deepcopy(y_test)

for i in range(len(temp)):
    temp[ i ] -= y_pred[ i ]
    temp[i] = abs(temp[i])
   



plt.grid(True)


df=pd.DataFrame({'x': range(len(y_pred)), 'y1': temp, 'y2': b.iloc[lower : upper , -1]})
#plt.grid(True) 

plt.plot( 'x', 'y1', data=df, color='red', linewidth=3, label = "Random forest")
plt.plot( 'x', 'y2', data=df, color='blue', linewidth=3, label = "Current Method")



# x-axis label 
plt.xlabel('x - axis (Over range)') 
# frequency label 
plt.ylabel('y - axis (Runs)') 
# plot title 
plt.title('Error Graph') 

#to print which line represent which values
plt.legend()

"""
#plt.plot(range(len(y_pred)) , temp , "red")
#plt.plot(range(len(y_pred)) , b.iloc[lower : upper , -1], "blue")

################################error compare#########################################


 






###############################compare current with our method############################ 

from sklearn import metrics  
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
print('RMSE current:',np.sqrt(metrics.mean_squared_error(y_test,b.iloc[lower : upper, -3]))) 
 

df=pd.DataFrame({'x': range(len(y_pred)), 'y1': y_pred, 'y2': b.iloc[lower : upper , -3], 'y3': y_test })
plt.grid(True) 

plt.plot( 'x', 'y1', data=df, color='red', linewidth=3, label = "Random forest")
plt.plot( 'x', 'y2', data=df, color='blue', linewidth=3, label = "Current Method")
plt.plot( 'x', 'y3', data=df, color='gold', linewidth=3, label = "Actual Runs")


# x-axis label 
plt.xlabel('x - axis (Over range)') 
# frequency label 
plt.ylabel('y - axis (Runs)') 
# plot title 
plt.title('Comparison Graph') 

#to print which line represent which values
plt.legend()



###############################compare current with our method############################




"""
import copy
random_forests = copy.deepcopy(temp)
multipl_reg = copy.deepcopy(temp)

plt.grid(True)    
plt.plot( range(len(y_pred)) , random_forests , "RED")
plt.plot(range(len(y_pred)) , multipl_reg  , "BLUE" )    
plt.plot( range(len(y_pred)) ,b.iloc[lower : upper , -1] , "yellow" )
"""




























