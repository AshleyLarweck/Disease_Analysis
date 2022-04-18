#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""HW3_skeleton.ipynb


#### % import necessary modules
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import errorbar, hist, scatter, xticks
from numpy import mean, std, sort, argsort, corrcoef, arange
from numpy.random import rand, randn, randint, choice, lognormal
from scipy.stats import norm, lognorm, expon, pareto
from math import sqrt
import scipy.stats as st
from scipy.stats.stats import pearsonr 



#%% load the data and setup some variables

### data is a numpy ndarray of shape (3,41,12) for the three diseases, 41 years and 12 months

diseases = ['Measles', 'Mumps', 'ChickenPox']
year = np.arange(1931, 1972)
month = np.arange(1, 13)
month_str = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

url = 'diseases.csv'
# uncomment the following to read directly from github
url = 'https://raw.githubusercontent.com/jianhuaruan/3753/main/' + url
data = pd.read_csv(url,header=None).values
data = data.reshape([3, 41, 12]) #data[0], data[1], and data[2] is for measles, mumps, and chickenpox

#%% Q1 calucate and show mean number of cases per year, and 95% CI of the mean"""

mean_Cases_Per_Year = data.mean(axis=2) 

col_Index = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41"]
mean_Cases_Per_Year = mean_Cases_Per_Year.T

std_Val = std(mean_Cases_Per_Year,0)
sqrt_Val = (sqrt(mean_Cases_Per_Year.shape[0]))
S_Error = std_Val / sqrt_Val

plt.figure()
plt.title('Fig 1: # of disease cases per year (mean & 95% CI)')

xticks([1,2,3], ["Measles", "Mumps", "ChickenPox"])

plt.errorbar([1,2,3], mean(mean_Cases_Per_Year,0), S_Error*1.96, marker='d', linestyle='', capsize=5)

plt.show()

#%% Q2 calucate and show percent of cases occurred in each month

plt.figure()
plt.title('Fig 2: Percent of cases in each month')

Measles_Monthly_Values = sum(data[0,:,:])
Measles_Total = (sum(Measles_Monthly_Values))
Measles_Monthly_Avg = (Measles_Monthly_Values/Measles_Total) *100

Mumps_Monthly_Values = sum(data[1,:,:])
Mumps_Total = sum(Mumps_Monthly_Values)
Mumps_Monthly_Avg = (Mumps_Monthly_Values/Mumps_Total) *100

CP_Monthly_Values = sum(data[2,:,:])
CP_Total = sum(CP_Monthly_Values)
CP_Monthly_Avg = (CP_Monthly_Values/CP_Total) *100

plt.xlabel('Months')
plt.ylabel('Percent of Cases')

plt.plot(Measles_Monthly_Avg, label = "Measles")
plt.plot(Mumps_Monthly_Avg, label = "Mumps")
plt.plot(CP_Monthly_Avg, label = "ChickenPox")
plt.legend()

plt.show()

#%% Q3 Scatter plot and correlation, mean monthly cases of Measles vs mumps"""

# Q3.1 scatter plot 

ind1 = 0
ind2 = 1

plt.figure()

def zscore(Measles_Monthly_Values): 
    return (Measles_Monthly_Values - mean(Measles_Monthly_Values))/std(Measles_Monthly_Values)


# Q3.2 Pearson correlation 
z_measles = zscore(Measles_Monthly_Values)
z_mumps = zscore(Mumps_Monthly_Values)
P_Coef = z_measles.dot(z_mumps) / len(Measles_Monthly_Values)

# Q3.2 Spearman correlation coefficient 
Measles_rank = argsort(argsort(Measles_Monthly_Values))
Mumps_rank = argsort(argsort(Mumps_Monthly_Values))
S_Coef = corrcoef(Measles_rank, Mumps_rank)[0,1]

plt.title('Fig 3: Mean monthly cases of %s vs %s' %(diseases[ind1], diseases[ind2]))
plt.text(2600, 350,'Pearson corr: %.4f ' %P_Coef)
plt.text(2600, 250,'Spearman corr: %.4f ' %S_Coef)


plt.scatter(Measles_Monthly_Values/40, Mumps_Monthly_Values/40)


plt.title('Fig 3: Mean Monthly Cases of Measles vs. Mumps')
plt.xlabel("Mean Monthly Cases of Measles")
plt.ylabel("Mean Monthly Cases of Mumps")

print('Pearson corr: %.4f ' %P_Coef)
print('Spearman corr: %.4f ' %S_Coef)

plt.show()

#%% Q4 Scatter plot and correlation, annual cases of measles vs mumps"""

# 4.1 scatter plot 

ind1 = 0
ind2 = 1

Annual_Values = data.sum(axis=2) 
Measles_Annual_Values = Annual_Values[0,:]
Mumps_Annual_Values = Annual_Values[1,:]
CP_Annual_Values = Annual_Values[2,:]

plt.figure()
plt.title('Fig 4: Annual cases of %s vs %s' %(diseases[ind1], diseases[ind2]))
plt.scatter(Measles_Annual_Values, Mumps_Annual_Values)

# 4.2 Pearson correlation coefficient 

z_measles_annual = zscore(Measles_Annual_Values)
z_mumps_annual = zscore(Mumps_Annual_Values)
P_Coef_annual = z_measles_annual.dot(z_mumps_annual) / len(Measles_Annual_Values)

# 4.3 Spearman correlation 
Measles_rank_annual = argsort(argsort(Measles_Annual_Values))
Mumps_rank_annual = argsort(argsort(Mumps_Annual_Values))
S_Coef_annual = corrcoef(Measles_rank_annual, Mumps_rank_annual)[0,1]

plt.title('Fig 3: Mean annual cases of %s vs %s' %(diseases[ind1], diseases[ind2]))

plt.scatter(Measles_Annual_Values, Mumps_Annual_Values)

plt.title('Fig 3: Mean Monthly Cases of Measles vs. Mumps')
plt.xlabel("Mean Monthly Cases of Measles")
plt.ylabel("Mean Monthly Cases of Mumps")

plt.figure()
plt.scatter(Measles_Annual_Values, Mumps_Annual_Values)

labels = np.zeros([41,1], dtype = int)
labels = ['1931','1951','1971']

plt.annotate(labels[0], xy=(Measles_Annual_Values[0], Mumps_Annual_Values[0]), xytext=(5, 0), textcoords='offset points')
plt.annotate(labels[1], xy=(Measles_Annual_Values[20], Mumps_Annual_Values[20]), xytext=(5, 0), textcoords='offset points')
plt.annotate(labels[2], xy=(Measles_Annual_Values[40], Mumps_Annual_Values[40]), xytext=(5, 0), textcoords='offset points')

plt.title("Annual Cases of Measles vs. Mumps")
plt.xlabel("Annual Cases of Measles")
plt.ylabel("Annual Cases of Mumps")

plt.text(52000, 4800,'Pearson corr: %.4f ' %P_Coef_annual)
plt.text(52000, 4000,'Spearman corr: %.4f ' %S_Coef_annual)

plt.show()

print('Pearson correlation: %.4f' %P_Coef_annual)
print('Spearman correlation: %.4f ' %S_Coef_annual)

#%% Q5 Scatter plot and correlation, monthly cases of Measles vs mumps

# 5.1 scatter plot and correlation in original space

Measles_Cases_By_Month = data[0,:,:]
Mumps_Cases_By_Month = data[1,:,:]


plt.figure()
plt.scatter(Measles_Cases_By_Month, Mumps_Cases_By_Month)

ind1 = 0
ind2 = 1

plt.title('Fig 5.1: Monthly cases of %s vs %s' %(diseases[ind1], diseases[ind2]))

# Pearson correlation coefficient 

Measles_reshaped=np.reshape(Measles_Cases_By_Month, (1, 492))

Mumps_reshaped=np.reshape(Mumps_Cases_By_Month,(1, 492))

P_Coef_Month=(np.corrcoef(Measles_reshaped,Mumps_reshaped))
P_Coef_Month = P_Coef_Month[0,1]

# Spearman correlation between annual cases of mumps vs chicken pox

Measles_rank_by_month = argsort(argsort(Measles_reshaped))
Mumps_rank_by_month = argsort(argsort(Mumps_reshaped))
S_Coef_by_month = corrcoef(Mumps_rank_by_month,Measles_rank_by_month)[1,0]

plt.text(17000, 480,'Pearson corr: %.4f ' %P_Coef_Month)
plt.text(17000, 300,'Spearman corr: %.4f ' %S_Coef_by_month)

plt.show()

print('Pearson correlation in orignal space: %.4f' %P_Coef_Month)
print('Spearman correlation in original space: %.4f ' %S_Coef_by_month)

# 5.2 scatter plot and correlation in log space

plt.figure()

ind1 = 0
ind2 = 1

plt.title('Fig 5.2: Monthly cases of %s vs %s' %(diseases[ind1], diseases[ind2]))

plt.yscale("log")  
plt.xscale("log") 

plt.scatter(Measles_Cases_By_Month, Mumps_Cases_By_Month)

plt.xlabel("Annual Cases of Measles")
plt.ylabel("Annual Cases of Mumps")

# Pearson correlation coefficient 
P_Coef_Month_log = (np.corrcoef(np.log(Measles_reshaped),np.log(Mumps_reshaped)))
P_Coef_Month_log = P_Coef_Month_log[0,1]

# Spearman correlation 

Measles_rank_by_month_log = argsort(argsort(np.log(Measles_reshaped)))
Mumps_rank_by_month_log = argsort(argsort(np.log(Mumps_reshaped)))
S_Coef_by_month_log = corrcoef(Measles_rank_by_month_log,Mumps_rank_by_month_log)[1,0]

plt.text(250, 70,'Pearson corr in log space: %.4f ' %P_Coef_Month)
plt.text(250, 50,'Spearman corr in log space: %.4f ' %S_Coef_by_month)

plt.show()

print('Pearson correlation in log space: %.4f' %P_Coef_Month_log)
print('Spearman correlation in log space: %.4f ' %S_Coef_by_month_log)

#%% Q6 (bonus) Correlation between number of mumps cases in different months"""

Matrix = np.corrcoef(Mumps_Cases_By_Month.T)

plt.imshow(Matrix)
plt.colorbar()

plt.title('Fig 6: Correlation between monthly Mumps cases' )
plt.xlabel("Month")
plt.ylabel("Month")

plt.xticks(range(12), range(1,13))
plt.yticks(range(12), range(1,13))

#%% Q7 (Bonus) calculate and show average perecent of diseases occurred in each month"""

Fij = np.zeros([3,12], dtype = float)

Fij[0,:] = Measles_Cases_By_Month.sum / Measles_Cases_By_Month.mean(axis=0)
Fij[1,:] = Mumps_Cases_By_Month.mean(axis=0) 
Fij[2,:] = data[1,:,:].mean(axis=0)



plt.xticks(range(12), range(1,13))


plt.xlabel('Months')
plt.ylabel('Percent of Cases')

plt.plot(Fij[0,:], label = "Measles")
plt.plot(Fij[1,:], label = "Mumps")
plt.plot(Fij[2,:], label = "ChickenPox")
plt.legend()


plt.show()
