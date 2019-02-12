import torch
from torch.nn import functional as F  

def train(model, train_loader, optimizer, loss, device, epoch):
    model.train()
    batches_n = len(train_loader)
    for batch_idx, batch_sample in enumerate(train_loader):
        rgb, data = batch_sample['image'].to(device), batch_sample['data'].to(device)
        target = batch_sample['target'].to(device)
        output = model(rgb, data)
        train_loss = loss(output, target)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #iteration = epoch * batches_n + batch_idx

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / batches_n, loss))