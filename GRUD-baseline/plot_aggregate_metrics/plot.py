import pickle_utils as pu
import matplotlib.pyplot as plt

data = pu.load('metrics.pkl.gz')

labels = []
for k in data:
    if 'tpr_ppv' not in k:
        continue
    x = list(data[k].keys())
    x.sort()
    y = list(data[k][i] for i in x)
    l, = plt.plot(x, y, label=k)
    labels.append(l)

plt.legend(labels)
plt.show()
