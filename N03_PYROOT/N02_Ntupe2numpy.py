import pandas as pd
from ROOT import *
from root_numpy import root2array, tree2array
from root_numpy import testdata
from IPython.display import display

# read signal dataset
sig_file = TFile.Open('signal.root') 
sig_tree = sig_file.Get('tree')   
sig_arr  = tree2array(sig_tree)	
sig_df   = pd.DataFrame(sig_arr)	
sig_df.to_csv('data.csv', mode='w',header=False)

#read background dataset
bkg_file = TFile.Open('background.root') 
bkg_tree = bkg_file.Get('tree')   
bkg_arr  = tree2array(bkg_tree)	
bkg_df   = pd.DataFrame(bkg_arr)	
bkg_df.to_csv('data.csv', mode='a',header=False)

display(sig_df)
display(bkg_df)


