from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot  as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Laden der Daten
Train_Data = datasets.MNIST('data',train = True, download = True, transform=transforms.ToTensor())
Test_Data = datasets.MNIST('data',train = False,download = True,transform=transforms.ToTensor())

# Erstellen der Batches für das Training
Batch_Train = DataLoader(Train_Data , batch_size= 32,shuffle = True)
Batch_Test = DataLoader(Test_Data, batch_size= 32,shuffle = False)

# Initialisierung des Netzwerkes
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,1,1),
            nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Sequential(
            nn.Linear(14*14*16,32),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(32,10))

    def forward(self,x):
        x = self.layer1(x)
        x = x.view(-1,14*14*16)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def training_loop(epochs, optimizer, model,loss_function, Train_Batch,Test_Batch):
    # Initialisierung der Variablen zur Bestimmung der Gemauigkeit und des Wertes der Verlustfunktion
    correct_array_percent_val = [0]
    correct_array_percent_train = [0] 
    loss_array_train = []
    loss_array_test = []
    
    # Schleife zum Trainieren des Netzwerkes und Bestimmung des Wertes der Verlustfunktion für die Trainingsdaten
    for i in range(1, epochs+1):
        loss_value_train = 0
        for image,label in Train_Batch:
            
            output = model(image)
            loss = loss_function(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value_train = loss_value_train + loss.item()
        
        loss_array_train.append(loss_value_train/len(Train_Batch))
        loss_value_test = 0
        
        # Schleife zur Berechnung des Wertes der Verlustfunktion für die Testdaten
        with torch.no_grad():
            for image,label in Test_Batch:
            
                out = model(image)
                loss = loss_function(out,label)
                loss_value_test = loss_value_test + loss.item()
            loss_array_test.append(loss_value_test/len(Test_Batch))
        
        # Schleife zur Bestimmung der relativen Genauigkeit des Netzwerkes
        for name, batch in [  ("train" , Train_Batch) , ("test" , Test_Batch) ]:
            correct = 0
            total = 0

            # Schleife zur Bestimmung der Anzahl der richtigen Klassifizierungen
            with torch.no_grad():
                for image, label in batch :
                    out = model(image)
                    _, predicted = torch.max(out, dim= 1)
                    total = total + label.shape[0]
                    correct = correct + int((predicted == label).sum() )
            
            # Ausgabe der Genauigkeit für jede Epoche, damit der Trainingsprozess überwacht werden kann
            print("Genauigkeit {}: {:.4f}".format(name, correct / total))
            
            if name == "test" :
                correct_array_percent_val.append(correct/total)
            if name == "train":
                correct_array_percent_train.append(correct/total)
    return correct_array_percent_val,correct_array_percent_train,loss_array_test,loss_array_train
        
epochen = [0,1,2,3,4,5,6,7,8,9,10]
x_loss = [1,2,3,4,5,6,7,8,9,10]

# Durchführung des Tests
model = CNN()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
relTest_SGD,relTrain_SGD,lTest_SGD,lTrain_SGD = training_loop(10,optimizer,model,nn.CrossEntropyLoss(),Batch_Train,Batch_Test)

model1 = CNN()
optimizer1 = torch.optim.SGD(model1.parameters(),lr=0.01, momentum=0.1)
relTest_ADA,relTrain_ADA,lTest_ADA,lTrain_ADA = training_loop(10,optimizer1,model1,nn.CrossEntropyLoss(),Batch_Train,Batch_Test)

model2 = CNN()
optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=0.001,eps=1e-7,alpha=0.9)
relTest_RMS,relTrain_RMS,lTest_RMS,lTrain_RMS = training_loop(10,optimizer2,model2,nn.CrossEntropyLoss(),Batch_Train,Batch_Test)

model3 = CNN()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.001,betas=(0.9,0.999), eps=1e-7)
relTest_ADAM,relTrain_ADAM,lTest_ADAM,lTrain_ADAM = training_loop(10,optimizer3,model3,nn.CrossEntropyLoss(),Batch_Train,Batch_Test)

# Visualisierung der Ergebnisse
plt.figure()
plt.plot(x_loss,lTrain_SGD,color = 'red')
plt.plot(x_loss,lTrain_ADA,color = 'green')
plt.plot(x_loss,lTrain_RMS, color = 'blue')
plt.plot(x_loss,lTrain_ADAM,color = 'pink')
plt.legend([ 'SGD','SGD mit Momentum','RMSprop','Adam' ],loc = 'upper right')
plt.title("Train loss")
plt.savefig("Loss_Train_MNIST.png")
plt.show()

plt.figure()
plt.plot(x_loss,lTest_SGD,color = 'red')
plt.plot(x_loss,lTest_ADA,color = 'green')
plt.plot(x_loss,lTest_RMS, color = 'blue')
plt.plot(x_loss,lTest_ADAM,color = 'pink')
plt.legend([ 'SGD','SGD mit Momentum','RMSprop','Adam' ],loc = 'upper right')
plt.title("Test loss")
plt.savefig("Loss_Test_MNIST.png")
plt.show()

plt.figure()
plt.plot(epochen,relTest_SGD,color = 'red')
plt.plot(epochen,relTest_ADA,color = 'green')
plt.plot(epochen,relTest_RMS, color = 'blue')
plt.plot(epochen,relTest_ADAM,color = 'pink')
plt.legend([ 'SGD','SGD mit Momentum','RMSprop','Adam' ],loc = 'lower right')
plt.title("Test Accuracy")
plt.savefig("Acc_Test_MNIST.png")
plt.show()

plt.figure()
plt.plot(epochen,relTrain_SGD,color = 'red')
plt.plot(epochen,relTrain_ADA,color = 'green')
plt.plot(epochen,relTrain_RMS, color = 'blue')
plt.plot(epochen,relTrain_ADAM,color = 'pink')
plt.legend([ 'SGD','SGD mit Momentum','RMSprop','Adam' ],loc = 'lower right')
plt.title("Train Accuracy")
plt.savefig("Acc_Train_MNIST.png")
plt.show()


