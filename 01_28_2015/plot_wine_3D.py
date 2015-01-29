from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
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

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

for label, marker, color in zip(
        range(1,4),('x', 'o', '^'),('blue', 'red', 'green')):

    ax.scatter(X_wine[:,0][y_wine == label],
                X_wine[:,1][y_wine == label],
                X_wine[:,2][y_wine == label],
                marker=marker,
                color=color,
                s=40,
                alpha=0.7,
                label='class {}'.format(label))

ax.set_xlabel('aclohol by volume in percent')
ax.set_ylabel('malic acid in g/L')
ax.set_zlabel('ash content in g/L')

plt.title('Wine dataset')

plt.show()
