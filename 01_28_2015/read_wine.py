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

# print some general information about the data
print('\ntotal number of samples (rows):', X_wine.shape[0])
print('total number of features (columns):', X_wine.shape[1])

# printing the first wine sample
# print formatting did not work..
float_formatter = lambda x: '{:.2f}'.format(x)
np.set_printoptions(formatter={'float_kind':float_formatter})
print('\n1st sample (i.e. 1st row):\nClass label: {:d}\n{:}\n'
            .format(int(y_wine[0]), X_wine[0]))

# printing the rel.frequency of the class labels
print('Class label frequencies')
print('Class 1 samples: {:.2%}'.format(float(list(y_wine).count(1))/float(y_wine.shape[0])))
print('Class 2 samples: {:.2%}'.format(float(list(y_wine).count(2))/float(y_wine.shape[0])))
print('Class 3 samples: {:.2%}'.format(float(list(y_wine).count(3))/float(y_wine.shape[0])))

return X_wine, y_wine
