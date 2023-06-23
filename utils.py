

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

def model_train(net, device, train_loader,  test_loader, optimizer, scheduler, criterion, train_set, val_set, val_loader):
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-05, amsgrad=False)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.01, epochs=20,steps_per_epoch=len(train_loader))
    loss_hist, acc_hist = [], []
    loss_hist_val, acc_hist_val = [], []

    for epoch in range(20):
        running_loss = 0.0
        correct = 0
        for data in train_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # compute training statistics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_set)
        avg_acc = correct / len(train_set)
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)

    # validation statistics
    net.eval()
    with torch.no_grad():
        loss_val = 0.0
        correct_val = 0
        for data in val_loader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)
            outputs = net(batch)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            loss_val += loss.item()
        avg_loss_val = loss_val / len(val_set)
        avg_acc_val = correct_val / len(val_set)
        loss_hist_val.append(avg_loss_val)
        acc_hist_val.append(avg_acc_val)
    net.train()

    scheduler.step(avg_loss_val)
    print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' % (epoch , avg_loss, avg_acc*100, avg_loss_val, avg_acc_val*100))





    legend = ['Train', 'Validation']
    plt.plot(loss_hist)
    plt.plot(loss_hist_val)
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(legend, loc='upper left')
    plt.show()

    legend = ['Train', 'Validation']
    plt.plot(acc_hist)
    plt.plot(acc_hist_val)
    lt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(legend, loc='upper left')
    plt.show()