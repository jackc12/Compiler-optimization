import pandas as pd, numpy as np, matplotlib.pyplot as plt

#Plot execution times
def plot_execution_times(predictions, actuals):
    times = [
        pd.DataFrame({
            'optimization': np.arange(predictions.shape[0]),
            'predicted': predictions[:,i],
            'actual': actuals[:,i]
        }) for i in range(5)
    ]
    figure, axes = plt.subplots(3, 2, figsize=(40,30))
    figure.delaxes(axes[2,1])
    times[0].plot.line(ax=axes[0,0], x='optimization', y=['actual', 'predicted'], color=['r','b']).set(title='execution time 1')
    times[1].plot.line(ax=axes[0,1], x='optimization', y=['actual', 'predicted'], color=['r','b']).set(title='execution time 2')
    times[2].plot.line(ax=axes[1,0], x='optimization', y=['actual', 'predicted'], color=['r','b']).set(title='execution time 3')
    times[3].plot.line(ax=axes[1,1], x='optimization', y=['actual', 'predicted'], color=['r','b']).set(title='execution time 4')
    times[4].plot.line(ax=axes[2,0], x='optimization', y=['actual', 'predicted'], color=['r','b']).set(title='execution time 5') 