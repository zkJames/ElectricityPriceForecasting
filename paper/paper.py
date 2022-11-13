import numpy as np
import matplotlib.pyplot as plt
trues = np.load("true.npy")
preds = np.load("pred.npy")
means = np.load("mean.npy")
stds = np.load("std.npy")
import torch
class StandardScaler():
    def __init__(self):
        self.mean = means[-1]
        self.std = stds[-1]

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

# plt.figure()
scaler = StandardScaler()
# plt.plot(scaler.inverse_transform(trues[0,:,-1]), label='GroundTruth')
# plt.plot(scaler.inverse_transform(preds[0,:,-1]), label='Prediction')
# plt.legend()
# plt.show()
print(scaler.inverse_transform(trues[20,:,-1]))

#138991 -- 167378