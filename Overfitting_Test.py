from torchvision import datasets
from torchvision import transforms
import torch as t
from matplotlib import pyplot  as plt

# Laden der Daten und Durchführung der Datenaugmentierung 
Train_Data= datasets.CIFAR10('data',train = True, download = True, transform=transforms.Compose([transforms.RandomRotation(degrees=10),transforms.RandomVerticalFlip(p=0.2), transforms.ToTensor()]))
Test_Data = datasets.CIFAR10('data',train = False,download = True,transform=transforms.ToTensor())

# Erstellen der Batches
train_Batch = t.utils.data.DataLoader(Train_Data , batch_size= 32,shuffle = True)
test_Batch = t.utils.data.DataLoader(Test_Data , batch_size= 32,shuffle = False)

# Definition des CNN, Conv. Layer erweitert um Batch-Norm. und Dropout, FC- Layer erweitert um Dropout und Batch-Norm.
class CNN(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,16,3,1,1), t.nn.BatchNorm2d(16),
            t.nn.ReLU(), t.nn.MaxPool2d(2), t.nn.Dropout(p=0.1))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(16,32,3,1,1), t.nn.BatchNorm2d(32),
            t.nn.ReLU(), t.nn.MaxPool2d(2), t.nn.Dropout(p=0.1))
        self.layer3 = t.nn.Sequential(
            t.nn.Conv2d(32,64,3,1,1), t.nn.BatchNorm2d(64),
            t.nn.ReLU(), t.nn.MaxPool2d(2), t.nn.Dropout(p=0.1))
        self.layer4 = t.nn.Sequential(
            t.nn.Conv2d(64,128,3,1,1), t.nn.BatchNorm2d(128),
            t.nn.ReLU(), t.nn.MaxPool2d(2), t.nn.Dropout(p=0.1))
        
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(2*2*128,32), t.nn.BatchNorm1d(32),
            t.nn.ReLU(), t.nn.Dropout(p=0.3))
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(32,10))

    def forward(self,x):
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)
        x = x.view(-1,2*2*128)
        x = self.fc1(x) 
        x = self.fc2(x)
        return x
    
# Funktion zum Trainieren und Testen
def training_loop(epochs, optimizer, model,loss_function, Batch_train,Batch_test):
    correct_array_percent_test = [0]
    correct_array_percent_train = [0] 
    loss_array_train = []
    loss_array_test = []
    for i in range(0, epochs):
        for image,label in Batch_train:
            
            outputs = model(image)
            loss = loss_function(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        for name, batch in [ ("train" , Batch_train) , ("test" , Batch_test) ]:
            correct = 0; total = 0 ;loss_value = 0
            with t.no_grad():
                for image, label in batch :
                    output = model(image)
                    
                    loss = loss_function(output,label)
                    loss_value = loss_value + loss.item()
                    
                    _, predicted = t.max(output, dim= 1)
                    total = total + label.shape[0]
                    correct = correct + int((predicted == label).sum() )
            print("Genauigkeit {}: {:.4f}".format(name, correct / total))
            
            if name == "train" :
                correct_array_percent_train.append(correct/total)
                loss_array_train.append(loss_value/len(Batch_train))
            if name == "test":
                correct_array_percent_test.append(correct/total)
                loss_array_test.append(loss_value/len(Batch_test))
    
    return correct_array_percent_test,correct_array_percent_train,loss_array_test,loss_array_train

# Initialisierung und Test 
model1 = CNN()
optimizer = t.optim.Adam(model1.parameters(), lr=0.001)
loss_function = t.nn.CrossEntropyLoss()
relTest1,relTrain1,lTest1,lTrain1 = training_loop(20,optimizer,model1,loss_function,train_Batch,test_Batch)


# Visualisierung der Daten
x_loss = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
epochen = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

plt.figure()
plt.plot(x_loss,lTrain1,color = 'black')
plt.plot(x_loss,lTest1,color = 'magenta')
#plt.plot(x_loss,lTrain2,color = 'red')
#plt.plot(x_loss,lTest2,color = 'red',linestyle = '--')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],[0,2,4,6,8,10,12,14,16,18,20])
plt.legend([ 'Training','Test'],loc = 'lower left')
plt.title("Verlauf der Verlustfunktion")
#plt.ylim(0,2)
plt.savefig("Loss_Overfitting_Test.png")
plt.show()

plt.figure()
plt.plot(epochen,relTrain1,color = 'black')
plt.plot(epochen,relTest1,color = 'magenta')
#plt.plot(epochen,relTrain2,color = 'red')
#plt.plot(epochen,relTest2,color = 'red',linestyle = '--')
plt.xticks([0,2,4,6,8,10,12,14,16,18,20],[0,2,4,6,8,10,12,14,16,18,20])
plt.legend([ 'Training','Test' ],loc = 'lower right')
plt.title("Relative Genauigkeit")
plt.ylim(0,1)
plt.savefig("Rel_Overfitting_Test.png")
plt.show()