

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error,make_scorer,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score

data=pd.read_csv("Tablet_press_data.csv")
print(data.head(10))
print("columns' names:",data.columns)
print("null:",data.isnull().sum())
print("describtion:", data.describe())
print("duplication",data.duplicated())
print("types:",data.dtypes)
data_dict={"Pressure":data["Pressure"].value_counts(),
      "Temperature":data["Temperature"].value_counts(),
      "Speed":data["Speed"].value_counts(),
      "Vibration":data["Vibration"].value_counts(),
      "Humidity":data["Humidity"].value_counts(),
      "Maintenance_Cycles":data["Maintenance_Cycles"].value_counts(),
      "Failure":data["Failure"].value_counts()
      }
print("value counts:",data_dict)


def plot(feature, kind):
    plt.figure(figsize=(8, 4))
    if kind == 'box' or kind == 'count':
        sns.catplot(x=feature, kind=kind, data=data, color='skyblue')
    elif kind == 'hist':
        sns.histplot(data=data, x=feature, color='skyblue')

    plt.title(f"{feature} Count Plot")
    plt.xlabel(f"{feature}")
    plt.ylabel("Total Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    if kind == 'count':
        ax = plt.gca()  # Get the current axis
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points')

    plt.show()




for feature in data:
    plot(feature, 'box')

for feature in data:
    plot(feature, 'hist')
correlation=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation , annot=True, edgecolor="red",fmt=".1", cbar=True)
plt.tight_layout()
plt.show()
x=pd.DataFrame(data, columns=['Pressure', 'Temperature', 'Speed', 'Vibration', 'Humidity',
       'Maintenance_Cycles'])
y=data["Failure"].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=45)
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print('Mean Absolute Error: ', mean_absolute_error(y_test, predictions))
print('Mean Squared Error: ', mean_squared_error(y_test, predictions))
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, predictions)))
print('R2 Score: ', r2_score(y_test, predictions))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_no_kfold = r2_score(y_test, predictions)
mse_no_kfold = mean_squared_error(y_test, predictions)
mae_no_kfold= mean_absolute_error(y_test, predictions)
rmse_no_kfold=np.sqrt(mean_squared_error(y_test, predictions))


r2_kfold = cross_val_score(model, x, y, cv=kf, scoring='r2').mean()
mse_kfold = -cross_val_score(model, x, y, cv=kf, scoring='neg_mean_squared_error').mean()
rmse_kfold = np.sqrt(-cross_val_score(model, x, y, cv=kf, scoring='neg_mean_squared_error').mean())
mae_kfold = -cross_val_score(model, x, y, cv=kf, scoring='neg_mean_absolute_error').mean()


print("r2_kfold=",r2_kfold)
print("mse_kfold=",mse_kfold)
print("rmse_kfold=",rmse_kfold)
print("mae_kfold=",mae_kfold)
metrics = ['R2 Score', 'MSE',"rmse","mae"]
before_kfold = [r2_no_kfold, mse_no_kfold,mae_no_kfold,rmse_no_kfold]
after_kfold = [r2_kfold, mse_kfold,rmse_kfold,mae_kfold]


X = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(X - width/2, before_kfold, width, label='No K-Fold', color='skyblue')
plt.bar(X + width/2, after_kfold, width, label='With K-Fold', color='orange')
plt.title('Comparison of Metrics: Before and After K-Fold', fontsize=14)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(X, metrics)
plt.ylim(0, max(max(before_kfold), max(after_kfold)) * 1.2)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()







