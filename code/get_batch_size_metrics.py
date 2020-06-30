import os, torch, pandas as pd
from sklearn.metrics import mean_squared_error
from find_token import find_token
from test_set import test_set
from net import net
net.eval()

batch_sizes, test_losses = [], []
for model in os.listdir('../diff_batch_sizes'):
    #Get batch size
    batch_size = int(find_token(model, 'batch_size=', '__lr='))
    
    #Load the model
    net.load_state_dict(torch.load('../diff_batch_sizes/' + model, map_location=torch.device('cpu')))
    
    test_predictions = []
    for start_index in range(0, test_set.shape[0] - batch_size + 1, batch_size):
        #get batches
        X = test_set[start_index:start_index + batch_size, :-5].view(batch_size,1,285)
        y = test_set[start_index:start_index + batch_size, -5:]
        #forward pass
        outputs = net(X).view(batch_size,5).detach()
        test_predictions.append(outputs)

    predictions = torch.cat(test_predictions, dim=0).numpy()
    
    
    
    batch_sizes.append(batch_size)
    test_losses.append(mean_squared_error(test_set.numpy()[:,-5:], predictions))

diff_batch_sizes = pd.DataFrame({
	'Batch Size': batch_sizes,
	'Test Loss': test_losses

})

diff_batch_sizes.sort_values(by='Batch Size').to_csv('../data/batch_size_metrics.csv')

