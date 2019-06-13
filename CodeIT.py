import pandas as pd # data structs and tools
import numpy as np #arrays matrices...
import matplotlib.pyplot as plt # plots, graphs,
import seaborn as sns # heat maps, time series , violin plots
import scipy as sp # integrals, diff eq, optimization
from sklearn.linear_model import LinearRegression #machine elarning!
import statsmodels # explore data, estimates statiistical models, stats test

pd.set_option('display.max_columns', 30) # allows larher display
'''
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
'''


#data aquisition: path(on computer, website, cloud)+ format(csv, json, excel, sql)
path='breast-cancer-wisconsin.data'
df=pd.read_csv(path, encoding="utf-8-sig",header=None)

#adding header
headers=["Sample code number","Clump Thickness","Uniformity of Cell Size",
   "Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size",
   "Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]                        
df.columns=headers

df.drop(labels="Sample code number",axis=1,inplace=True) # drop entire column


#info about attributes
'''
   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
'''
############Data preprosessing/wrangling/cleaning###################

#1.Handle missing values- N/a, ?, 0 blank cell______________________

    #a: drop missing value
df.replace(to_replace='?',value=np.nan,inplace=True)
df.dropna(axis=0, inplace=True)
#drops entire row with NaN , axis=1 drop entire column

#2.Data Handling/Formatting________________________________________
df["Bare Nuclei"]=df["Bare Nuclei"].astype("int")

#3.Data Normalization (scaling, centering)___________________________

#many more possiblilities: binning, categories

#################(EDA) exploratory data analysis###########################

#figure1=plt.scatter(df["Mitoses"],df["Clump Thickness"])
'''figure1=sns.boxplot(x="Class",y="Clump Thickness",data=df)
plt.show()'''

#Correlation Stats________________________________________________

    #pearsons correlation, lower p-value, better is certainty
print("Pearson coef | Pval")
for val in headers[1:]:
    Pearson_coef,p_val=sp.stats.pearsonr(df[val],df['Class'])
    print(val,Pearson_coef,'|',p_val)

x = df["Bare Nuclei"]
y = df["Uniformity of Cell Shape"]

col = np.where(df["Class"]==2,'k','r')
'''
plt.scatter(x, y, c=col, s=5, linewidth=0)
plt.show()'''
##################Model Development#####################################
#Regression data should be continuous
'''
#data aquisition: path(on computer, website, cloud)+ format(csv, json, excel, sql)
url='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
df=pd.read_csv(url, encoding="utf-8-sig",header=None)

#adding header
headers=["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base",
        "length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke",
        "compression-ratio","horse-power","peak-rpm","city-mpg","highway-mpg","price"]
df.columns=headers

y=df["highway-mpg"]
x=df["price"]
#figure1=plt.scatter(x,y)
#plt.show()

#Scratch code use np matrices y=XXT(theta)--> normal eq
#alternatively gradient descent theta=theta-(alpha)*d(cost)/d(theta)

#Sklearn
lm=LinearRegression()
X=df[['highway-mpg']]
Y=df['price']
lm.fit(X,Y)
linear_inter=lm.intercept_
linear_coef=lm.coef_
x_predict=28
y_predict=linear_inter+linear_coef*x_predict
print(y_predict)
print(lm.score(X,Y)) #R-squared 
'''


#clusters

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
df["Class"], df[""], random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)

knn.fit()










#neural networks



