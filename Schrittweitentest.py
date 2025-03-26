from torchvision import datasets
from torchvision import transforms
import torch as t
from matplotlib import pyplot  as plt

# Laden der Daten
Train_Data= datasets.CIFAR10('data',train = True, download = True, transform=transforms.ToTensor())
Test_Data = datasets.CIFAR10('data',train = False,download = True,transform=transforms.ToTensor())

# Erstellen der Batches
train_Batch= t.utils.data.DataLoader(Train_Data , batch_size= 32,shuffle = True)
test_Batch = t.utils.data.DataLoader(Test_Data , batch_size= 32,shuffle = False)

# Definition des CNN
class CNN(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,6,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(6,16,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(8*8*16,32),
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(32,10))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,8*8*16)

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
                loss_array_train.append(loss_value/len(train_Batch))
            if name == "test":
                correct_array_percent_test.append(correct/total)
                loss_array_test.append(loss_value/len(test_Batch))
    
    return correct_array_percent_test,correct_array_percent_train,loss_array_test,loss_array_train

# Initialisierung und Durchführung des Tests
model1 = CNN()
optimizer = t.optim.Adam(model1.parameters(), lr=0.1)
loss_function = t.nn.CrossEntropyLoss()
relTest1,relTrain1,lTest1,lTrain1 = training_loop(10,optimizer,model1,loss_function,train_Batch,test_Batch)

model2 = CNN()
optimizer = t.optim.Adam(model2.parameters(), lr=0.01)
loss_function = t.nn.CrossEntropyLoss()
relTest2,relTrain2,lTest2,lTrain2 = training_loop(10,optimizer,model2,loss_function,train_Batch,test_Batch)

model3 = CNN()
optimizer = t.optim.Adam(model3.parameters(), lr=0.001)
loss_function = t.nn.CrossEntropyLoss()
relTest3,relTrain3,lTest3,lTrain3 = training_loop(10,optimizer,model3,loss_function,train_Batch,test_Batch)

model4 = CNN()
optimizer = t.optim.Adam(model4.parameters(), lr=0.0001)
loss_function = t.nn.CrossEntropyLoss()
relTest4,relTrain4,lTest4,lTrain4 = training_loop(10,optimizer,model4,loss_function,train_Batch,test_Batch)

# Visualisierung der Ergebnisse
x_loss = [1,2,3,4,5,6,7,8,9,10]
epochen = [0,1,2,3,4,5,6,7,8,9,10]

plt.figure()
plt.plot(x_loss,lTrain1,color = 'purple')
plt.plot(x_loss,lTrain2,color = 'black')
plt.plot(x_loss,lTrain3,color = 'red')
plt.plot(x_loss,lTrain4,color = 'blue')
plt.legend([ 'alpha = 0.1','alpha = 0.01','alpha = 0.001','alpha  = 0.0001'],loc = 'lower left')
plt.title("Verlauf der Verlustfunktion")
plt.ylim(0,2.5)
plt.savefig("Loss_Schrittweitentest.png")
plt.show()

plt.figure()
plt.plot(epochen,relTrain1,color = 'purple')
plt.plot(epochen,relTrain2,color = 'black')
plt.plot(epochen,relTrain3,color = 'red')
plt.plot(epochen,relTrain4,color = 'blue')
plt.legend([ 'alpha = 0.1','alpha = 0.01','alpha = 0.001','alpha  = 0.0001' ],loc = 'upper left')
plt.title("Relative Genauigkeit")
plt.ylim(0,1)
plt.savefig("Rel_Schrittweitentest.png")
plt.show()