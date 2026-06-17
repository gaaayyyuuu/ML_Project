import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
df = pd.read_csv("covid-liver.csv")
features = ["Year","Age","Gender","Month","Cirrhosis","Size","PS"]
df["Year"] = df["Year"].map({"Prepandemic ": 0, "Postpandemic": 1}) 
df["Cirrhosis"] = df["Cirrhosis"].map({"Y": 0, "N": 1}) 
df["Gender"] = df["Gender"].map({"M": 0, "F": 1})
df["Size"] = df["Size"].fillna(df["Size"].median())
df=df.fillna(0)
y = df["Cancer"]
x = df[features]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
models = {"Logistic Regression": LogisticRegression(max_iter = 200),
          "Decision Tree": DecisionTreeClassifier(),
          "Random Forest": RandomForestClassifier(),
          "K-Nearest Neighbors": KNeighborsClassifier(),
          "Support Vector Machine": SVC()
          }
names = []
accuracy = []
for name,model in models.items():
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test,pred)
    accuracy.append(acc)
    names.append(name)
    print(name,":",acc)
plt.bar(names,accuracy,color = "blue")
plt.ylim(0,1)
plt.xlabel("Models")
plt.ylabel("Accuracies")
plt.title("Models Comparison")
plt.xticks(rotation = 20)
plt.show()
