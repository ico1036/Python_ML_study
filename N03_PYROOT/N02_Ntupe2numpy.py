import pandas as pd
from ROOT import *
from root_numpy import root2array, tree2array
from root_numpy import testdata
from IPython.display import display
import numpy as np



# --read signal dataset
sig_file = TFile.Open('signal.root') 
sig_tree = sig_file.Get('tree')   
sig_arr  = tree2array(sig_tree)	
sig_df   = pd.DataFrame(sig_arr)	

# --read background dataset
bkg_file = TFile.Open('background.root') 
bkg_tree = bkg_file.Get('tree')   
bkg_arr  = tree2array(bkg_tree)	
bkg_df   = pd.DataFrame(bkg_arr)	

# --Normalize
def MinMaxScaler(data):
    numerator = data - np.min(data,0)
    denominator = np.max(data,0) - np.min(data,0)
    return numerator / (denominator+1e-7)

norm=False
if norm == True:
	sig_df.iloc[:,:-1] = MinMaxScaler(sig_df.iloc[:,:-1])
	bkg_df.iloc[:,:-1] = MinMaxScaler(bkg_df.iloc[:,:-1])


# --Merge signal and bkg dataset
data = pd.concat([sig_df,bkg_df],ignore_index=True)

display(data)

data = data.values

# --Shuffle and Split dataset: Trainig, Validation, Test
inds = np.arange(data.shape[0])
tr   = int(0.6 * data.shape[0])  # Split ratio --> 6 : 2: 2
np.random.RandomState(11).shuffle(inds)

train_data = data[inds[:tr]]
rest_data   = data[inds[tr:]]
val_data = rest_data[:int(rest_data.shape[0] /2)]
test_data = rest_data[int(rest_data.shape[0] /2):]

print(data)
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)


# --Write data as csv format
np.savetxt('train_data.csv',train_data,fmt='%5.5f',delimiter =',')
np.savetxt('val_data.csv',val_data,fmt='%5.5f',delimiter =',')
np.savetxt('test_data.csv',test_data,fmt='%5.5f',delimiter =',')

