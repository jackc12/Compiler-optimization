from net import net
import torch

def predict(net, batch_size, test_set):
    test_predictions = []
    for start_index in range(0, test_set.shape[0] - batch_size + 1, batch_size):
        #get batches
        X = test_set[start_index:start_index + batch_size, :-5].view(batch_size,1,285)
        y = test_set[start_index:start_index + batch_size, -5:]
        #forward pass
        outputs = net(X).view(batch_size,5).detach()
        test_predictions.append(outputs)

    return torch.cat(test_predictions, dim=0).numpy()