import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv("covid-liver.csv")
print(df.head())
print(df.tail())
print(df.columns)
print(df.info())
features = ["Year","Bleed","Age","Gender","Month","Cirrhosis","Size","PS"]
df["Year"] = df["Year"].map({"Prepandemic ": 0, "Postpandemic": 1}) 
df["Cirrhosis"] = df["Cirrhosis"].map({"Y": 0, "N": 1}) 
df["Bleed"] = df["Bleed"].map({"Y": 0, "N": 1}) 
df["Gender"] = df["Gender"].map({"M": 0, "F": 1})
y = df["Cancer"]
x = df[features]
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(pred)
accuracy = accuracy_score(pred,y_test)
print(accuracy)
plt.bar(["Decision tree"],[accuracy], color = "green")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Decision Tree Classifier Based On Covid-19")
plt.show()


