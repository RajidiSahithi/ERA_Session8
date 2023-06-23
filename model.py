import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# Model for Session 8
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        drop=0.01
        BN=1
        GN=0
        LN=0
        if (BN == 1):
            self.conv1=  nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv23=  nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
            )

            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv56 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(drop)
            )

            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
            )

        elif (GN == 1):
            self.conv1=  nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(2,8),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv23=  nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(2,8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(2,8),
            nn.ReLU(),
            nn.Dropout(drop)
            )

            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(2,16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(2,32),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv56 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(2,32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(2,32),
            nn.ReLU(),
            nn.Dropout(drop)
            )

            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(2,16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(2,16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(2,8),
            nn.ReLU(),
            nn.Dropout(drop)
            )

        else:
            self.conv1=  nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1,8),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv23=  nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1,8),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(1,8),
            nn.ReLU(),
            nn.Dropout(drop)
            )

            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv3 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv4 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv56 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.GroupNorm(1,32),
            nn.ReLU(),
            nn.Dropout(drop)
            )

            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv7 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv8 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(),
            nn.Dropout(drop)
            )
            self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.GroupNorm(1,8),
            nn.ReLU(),
            nn.Dropout(drop)
            )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv10 = nn.Sequential(
        nn.Conv2d(in_channels=8, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # output  RF: 28
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv23(out) + out
        out = self.pool1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv56(out) + out
        out = self.pool1(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        out = self.gap(out)
        out = self.conv10(out)
        out = out.view(-1, 10)
        return out


def model_train(net, device, train_loader,  test_loader, optimizer, scheduler, criterion, train_set, val_set, val_loader):
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)
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

# Model for Session 7 - Initail model MODEL_1
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    #INPUT BLOCK
    self.conv1= self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
    #CONVOLUTIONAL BLOCK 1
    self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        )
    self.maxpool1=nn.MaxPool2d(2,2)
    #TRANSITION BLOCK 1 using kernal size 1 X 1 (Antman)
    self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU()
        )
    # CONVOLUTION BLOCK 2
    self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        )
    #self.maxpool2=nn.MaxPool2d(2,2)

    # CONVOLUTIONAL BLOCK
    self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(6, 6), padding=0, bias=False),

        )

  def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.maxpool1(x)
      x = self.conv3(x)
      x = self.conv4(x)
      #x = self.maxpool2(x)
      x = self.conv5(x)
      x = x.view(-1, 16)
       
      y = F.log_softmax(x,dim=-1)
      return y

# Model for Session 7 - Intermediate  model MODEL_2
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    drop=0.01
    #INPUT BLOCK
    self.conv1= self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    #CONVOLUTIONAL BLOCK 1
    self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    self.maxpool1=nn.MaxPool2d(2,2)
    #TRANSITION BLOCK 1
    self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            
        )
    # CONVOLUTION BLOCK 2
    self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    #self.maxpool2=nn.MaxPool2d(2,2)

    # GAP - Adaptive Global Average Pooling
    self.gap = nn.AdaptiveAvgPool2d(1)

    self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

  def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.maxpool1(x)
      x = self.conv3(x)
      x = self.conv4(x)
      #x = self.maxpool2(x)
      x = self.gap(x)
      x = self.conv5(x)
      x = x.view(-1, 10)
      return F.log_softmax(x,dim=-1)

# Model for Session 7 - BEST  model MODEL_3

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    drop=0.01
    #INPUT BLOCK
    self.conv1= self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    #CONVOLUTIONAL BLOCK 1
    self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop)
        )
    self.maxpool1=nn.MaxPool2d(2,2)
    #TRANSITION BLOCK 1
    self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(8),
            
        )
    # CONVOLUTION BLOCK 2
    self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=12, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(drop)
        )

    # GAP - Adaptive Global Average Pooling
    self.gap = nn.AdaptiveAvgPool2d(1)

    self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(1, 1), padding=0, bias=False), # output_size = 1    RF: 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),  # output  RF: 28
        )
        
             
# Model for Session 6

class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.network = nn.Sequential(
        #  layer 1
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(), #  feature map size = (28, 28)
        nn.BatchNorm2d(8),
        nn.Dropout(p=0),
        #  layer 2
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(8),
        nn.Dropout(p=0),

        nn.MaxPool2d(2), #  feature map size = (14, 14)
        #  layer 3
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(), #  feature map size = (14, 14)
        nn.BatchNorm2d(16),
        nn.Dropout(p=0),
        #  layer 4
        nn.Conv2d(16, 16, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(p=0),
        nn.MaxPool2d(2), #  feature map size = (7, 7)
        #  layer 5
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(), #  feature map size = (7, 7)
        nn.BatchNorm2d(32),
        nn.Dropout(p=0),
        #  layer 6
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(p=0),
        nn.MaxPool2d(2), #  feature map size = (3, 3)
        #  output layer
        nn.Conv2d(32, 10, 1),
        nn.AvgPool2d(3)
    )

  def forward(self, x):
    x = x.view(-1, 1, 28, 28)
    x = self.network(x)
    x = x.view(-1, 10)
     
    y = F.log_softmax(x,dim=-1)
    return y

def model_summary(model,input_size):
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))
    return model,input_size                            
     