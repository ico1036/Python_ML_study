import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

y_pred = np.load('prediction_nn_log.pyc.npy')
test_data = np.loadtxt('test_data.csv',delimiter=',')
y_true = test_data[:,-1]

idx_sig = np.where(y_true == 1)[0]
idx_bkg =np.where(y_true == 0)[0]

hist_pred_sig = y_pred[idx_sig]
hist_pred_bkg = y_pred[idx_bkg]


plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

bins = np.linspace(0,1,100)
plt.hist(hist_pred_sig,bins=bins,color='r',alpha=0.7)
plt.hist(hist_pred_bkg,bins=bins,color='b',alpha=0.7)
plt.savefig("score.png")
