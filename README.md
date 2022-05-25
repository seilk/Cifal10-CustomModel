# Cifal10-CustomModel
Kaggle : https://www.kaggle.com/competitions/ycs1003-cifar-10-competition-2022-1/leaderboard
- Test Acc : 90%
- LeaderBoard : 19/178

## Detail
- Torch
- Eviroment on Colab(Free ver.)
- I tried only to improve **accuracy** regardless of testing time.
- I did not use pre-trained model (i.e ResNet, VGG, etc)
### Data Augmentation
```python
transform_train = T.Compose([T.RandomAffine(30),
                            T.ColorJitter(),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
### CNN
- Using nn.Sequential() 
  - ref : https://michigusa-nlp.tistory.com/26
- Using filter size to retain input size at output 
  - Height and Width = 3, Stride = 1, Padding = 1
- BatchNormalization after every Conv2d
- ReLU
- Dropout
  - It does better when after ReLU 
- MaxPool2d
  - Finally original image shrink from 32\*32 to 2\*2
```python
class SimpleCNN(nn.Module) :

	def __init__(self) :
		super().__init__()

		# W, H = (N - F + 2P) / stride + 1
		# initial image size = 32 * 32

		self.conv_layers = nn.Sequential(

			# 3 32 32
			nn.Conv2d(3, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
			nn.Dropout2d(0.3),
			nn.Conv2d(32, 32, 3, 1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),

			nn.Conv2d(32, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			nn.Dropout2d(0.4),
			nn.Conv2d(64, 64, 3, 1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
			nn.MaxPool2d(2, 2),

			# 64 16 16
			nn.Conv2d(64, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			nn.Dropout2d(0.5),
			nn.Conv2d(128, 128, 3, 1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
			nn.MaxPool2d(2, 2),

			# 128 8 8
			nn.Conv2d(128, 256, 3, 1,padding = 1), nn.BatchNorm2d(256), nn.ReLU(),
			nn.Dropout2d(0.6),
			nn.Conv2d(256, 256, 3, 1,padding = 1), nn.BatchNorm2d(256), nn.ReLU(),
			nn.MaxPool2d(2, 2),

			#256 4 4
			nn.Conv2d(256, 512, 3, 1,padding = 1), nn.BatchNorm2d(512), nn.ReLU(),
			nn.Dropout2d(0.7),
			nn.Conv2d(512, 512, 3, 1,padding = 1), nn.BatchNorm2d(512), nn.ReLU(),
			nn.MaxPool2d(2, 2),
		)
  
		self.fc = nn.Sequential(
			nn.ReLU(),
			nn.Linear(512*2*2, 10, bias=True),
		)

		# self.initialize_weights()

	def forward(self, x) :
		features = self.conv_layers(x)
		flatten = torch.flatten(features, 1)  # flatten
		x = self.fc(flatten)
		return x
```
### HyperParams
```python
batch_size = 512 # not relate to Overfitting
learning_rate = 0.008
num_epochs = 1000
```
### Training 
```python
from statistics import mean 
import sys 

def train(optimizer, model, num_epochs=10, first_epoch=1):
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []

    for epoch in range(first_epoch, first_epoch + num_epochs):
        print('Epoch', epoch)
        
        # train phase
        model.train()
        # create a progress bar
        progress = ProgressMonitor(length=len(train_set))
        
        # keep track of predictions
        correct_train = 0

        batch_losses = []

        # scheduler.step()  # ! lr scheduler
        
        for batch, targets in train_loader:
            
            # Move the training data to the GPU
            batch = batch.to(device)
            targets = targets.to(device)

            # clear previous gradient computation
            optimizer.zero_grad()

            # forward propagation
            outputs = model(batch)

            # calculate the loss
            loss = criterion(outputs, targets)

            # backpropagate to compute gradients
            loss.backward()

            # update model weights
            optimizer.step()

            batch_losses.append(loss.item())

            # accumulate correct count
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == targets.data)

            # update progress bar
            progress.update(batch.shape[0], mean(batch_losses) )  

        # scheduler.step()
        train_losses.append( mean(batch_losses))


        # test phase
        model.eval()

        y_pred = []

        correct_test = 0


        # We don't need gradients for test, so wrap in 
        # no_grad to save memory
        with torch.no_grad():

            for batch, targets in test_loader:

                # Move the training batch to the GPU
                batch = batch.to(device)
                targets = targets.to(device)

                # forward propagation
                outputs = model(batch)

                # calculate the loss
                loss = criterion(outputs, targets)

                # save predictions
                y_pred.extend( outputs.argmax(dim=1).cpu().numpy() )

                # accumulate correct count
                _, preds = torch.max(outputs, 1)
                correct_test += torch.sum(preds == targets.data)
                

        # Calculate accuracy
        train_acc = correct_train.item() / train_set.data.shape[0]
        test_acc = correct_test.item() / test_set.data.shape[0]

        print('Training accuracy: {:.2f}%'.format(float(train_acc) * 100))
        print('Test accuracy: {:.2f}%\n'.format(float(test_acc) * 100))
        
        # scheduler.step()
    
    return train_losses, test_losses, y_pred
```
