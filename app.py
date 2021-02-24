
import tkinter as tk
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv(
    "insurance.csv")

# replaced the strings in datasets with some values
df = df.replace('female', 1)
df = df.replace('male', 2)
df = df.replace('yes', 1)
df = df.replace('no', 0)

# taking the columns required in the dataset
cdf = df[['age', 'sex', 'children', 'bmi', 'smoker', 'charges']]

print(cdf.head())

# checked the graphical form of data to choose a suitable algorithm
# plt.scatter(cdf.bmi, cdf.charges,  color='blue')
# plt.xlabel("bmi")
# plt.ylabel("charges")
# plt.show()

# dividing the data in 8:2 ratio: 80% used for training and rest for testing
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# applying linear regression
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['age', 'sex', 'children', 'bmi', 'smoker']])
y = np.asanyarray(train[['charges']])
regr.fit(x, y)

# The coefficients
print('Coefficients: ', regr.coef_)

# y_hat = regr.predict(test[['age', 'sex', 'children', 'bmi', 'smoker']])
# print(y_hat[0])

x = np.asanyarray(test[['age', 'sex', 'children', 'bmi', 'smoker']])
y = np.asanyarray(test[['charges']])
# print("Residual sum of squares: %.2f"
#   % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))


# creating the ui using tkinter


fields = 'Age', 'Sex (1 for female and 2 for male)', 'Children', 'Bmi', 'Smoker (Yes 1 No 0)'

arr = []

#fetching data from user and adding them to the array "arr"
def fetch(entries):
    for entry in entries:
        field = entry[0]
        text = entry[1].get()
        arr.append(float(text))
        print('%s: "%s"' % (field, text))
    print(arr)
    t = "Predicted cost: " + str(output(arr))
    w = tk.Label(root, text=t)
    w.pack()

#creating GUI in tkinter
def makeform(root, fields):
    root.title("Medical Insurance Cost Predictor")
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=30, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

#predicting the cost
def output(arr):
    y_hat = regr.predict(np.asanyarray([arr]))
    return y_hat[0]

#main loop
if __name__ == "__main__":
    root = tk.Tk()

    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    b1 = tk.Button(root, text='Show',
                   command=(lambda e=ents: fetch(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    # w = tk.Label(root, text=arr[2])
    # w.pack()

    root.mainloop()
