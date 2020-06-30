import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


from net import net
losses = []
net = net.to(device)
net.train()


#Set hyperparameters
batch_size = 32
epochs = 500
lr=1e-4
weight_decay = 1e-4

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


#Get Train set
from train_set import train_set
train_set = train_set.to(device)

#Training loop
for epoch in range(epochs):
    
    #Set loss to zero
    running_loss = 0.0
    
    #Shuffle training set
    shuffled_train_set = train_set[torch.randperm(train_set.shape[0])]
    
    for start_index in range(0, shuffled_train_set.shape[0] - batch_size, batch_size):
        
        #get batches
        X = shuffled_train_set[start_index:start_index + batch_size, :-5].view(batch_size,1,285)
        y = shuffled_train_set[start_index:start_index + batch_size, -5:]
        
        #zero gradients
        optimizer.zero_grad()
        
        #forward pass
        outputs = net(X).view(batch_size,5)
        
        #get loss
        loss = criterion(outputs, y)
        
        #backward pass
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if epoch % 50 == 49 or epoch == 0:    # print every 10 epochs
        print('{} epoch {} loss: {}'.format('', epoch if epoch > len(losses) else len(losses), running_loss))
    losses.append(running_loss)


from plot_losses import plot_losses
plot_losses(losses)


from sklearn.metrics import mean_squared_error

from predict import predict
from plot_execution_times import plot_execution_times
from test_set import test_set
test_set = test_set.to(device)
net.eval()


predictions = predict(net, 32, test_set)

print('Test Loss:', mean_squared_error(test_set[:,-5:], predictions))
plot_execution_times(predictions, test_set[:, -5:])