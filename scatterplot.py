# -*- coding: utf-8 -*-



import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset("tips")
df=tips.copy()


sns.scatterplot(x ="total_bill" , y ="tip", hue ="day", size="size", data=df)
plt.show()

sns.lmplot(x="total_bill", y="tip", hue="smoker", col ="time" ,row ="sex", data=df)
plt.show()



import seaborn as sns
import matplotlib.pyplot as plt


iris = sns.load_dataset("iris")
df=iris.copy()
print(df.head())

sns.pairplot(df , kind="reg" ,hue="species"  )