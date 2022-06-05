from matplotlib import pyplot as plt

# All models that we have used
x = ['Linear Regression', 'XGBoost', 'Decision Tree', 'Random Forest']
# These are the errors we are getting in respective model
# (For XGB and RF error keeps changing in every run - so these errors are when we run the program)
y = [22517.43, 19331.86, 24738.96, 18578.86]

colors = ["#c05949", "#d86f3d", "#e88b2b", "#edab06"]

fig, ax = plt.subplots()
plt.barh(y=range(len(x)), tick_label=x, width=y, height=0.4, color=colors)
ax.set(xlabel="MAE (Lower is better)", ylabel="Model")
plt.show()