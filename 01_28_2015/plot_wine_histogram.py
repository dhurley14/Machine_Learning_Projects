from matplotlib import pyplot as plt
from math import floor, ceil
import numpy as np

all_data = np.loadtxt(open("./wine_data.csv","r"),
        delimiter=',',
        skiprows=0,
        dtype=np.float64
        )

# for each row (:) load the class labels (0)
y_wine = all_data[:,0]

# conversion of the class labels to integer-type array
y_wine = y_wine.astype(np.int64, copy=False)

# load the 14 features (i.e. for all rows, load data from column 1 -> end)
X_wine= all_data[:,1:]


plt.figure(figsize=(10,8))

# bin width of the histogram in steps of 0.15
bins = np.arange(floor(min(X_wine[:,0])), ceil(max(X_wine[:,0])), 0.15)
#print(bins)

# get the max count for a particular bin for all classes combined
max_bin = max(np.histogram(X_wine[:,0], bins=bins)[0])

colors = ('blue', 'red', 'green')

for label,color in zip(
        range(1,4), colors):

    mean = np.mean(X_wine[:,0][y_wine == label])# class sample mean
    stdev = np.std(X_wine[:,0][y_wine == label])
    plt.hist(X_wine[:,0][y_wine == label],
            bins=bins,
            alpha=0.3, # opacity level
            label='class {} ($\mu={:.2f}$, $\sigma={:.2f}$)'.format(label, mean, stdev),
            color=color)

plt.ylim([0, max_bin*1.3])
plt.title('Wine data set - Distribution of alcohol contents')
plt.xlabel('alcohol by volume', fontsize=14)
plt.ylabel('count', fontsize=14)
plt.legend(loc='upper right')

plt.show()
