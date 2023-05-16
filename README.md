# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.
# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
# ALGORITHM
# STEP 1
Read the given Data
# STEP 2
Clean the Data Set using Data Cleaning Process
# STEP 3
Apply Feature Transformation techniques to all the features of the data set
# STEP 4
Save the data to the file
# CODE
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
df
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()
df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()
df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()
df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUPUT
# Dataset:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/8ad47618-5e0f-4bb4-a64e-27710bd3d5a0)
# Head:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/64530e05-00d8-4439-b208-9a6e8e333925)
# Null data:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/084587a0-ebfb-4698-b3b4-b9e0188368cd)
# Information:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/02784664-0081-4156-81fd-13678de01041)
# Description:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/ac4735b6-5859-4e4e-a7d2-9225477e212d)
# Highly Positive Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/6132cf20-174f-409b-b8f7-106f6efc3de7)
# Highly Negative Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/707a052c-5283-46c7-8c36-87b3cb8fec0e)
# Moderate Positive Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/a4185a18-2143-479f-8cb2-e8a01aef81c1)
# Moderate Negative Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/d9010520-5569-4eb6-ab62-7f7cc764b4e8)
# Log of Highly Positive Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/f1e441ad-f395-48dd-b7b2-5a964590af58)
# Log of Moderate Positive Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/00ac7d91-622d-477f-8210-771e9a8e6909)
# Reciprocal of Highly Positive Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/31a0af3d-7168-47d9-8aaf-e9047e092b0f)
# Square root tranformation:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/d317044d-edaa-482c-8925-32a207fc6440)
# Power transformation of Moderate Positive Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/17ba517c-6f18-45cc-ad88-b787736b518f)
# Power transformation of Moderate Negative Skew:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/10ac0ac8-1a8d-4d18-9ec0-2d2bd41b79e3)
# Quantile transformation:
![image](https://github.com/Sahithya7/Ex-06-Feature-Transformation/assets/133002193/31f9b21e-f457-4637-b6b2-e990bb0b81d6)
# RESULT:
Thus, Feature transformation is performed and executed successfully for the given dataset
