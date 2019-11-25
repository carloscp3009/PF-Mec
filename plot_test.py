import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = np.genfromtxt('Global_idx.csv', delimiter=',') 


# Plot a filled kernel density estimate
sns.lmplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])


plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()