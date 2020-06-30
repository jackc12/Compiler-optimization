import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Plot execution times
def plot_losses(losses):
    pd.DataFrame({
    	'epochs': np.arange(500),
    	'losses': losses[-500:]
    }).plot.line(x='epochs', y='losses', color=['r','b'], figsize=(20, 10)).set(title='last 500 losses', xlabel='epochs', ylabel='loss')