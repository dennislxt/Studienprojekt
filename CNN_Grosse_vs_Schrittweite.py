# Importieren der Bibliotheken
from torchvision import datasets
from torchvision import transforms
import torch as t
from matplotlib import pyplot  as plt

# Laden der Datensätze
Train_Data= datasets.CIFAR10('data',train = True, download = True, transform=transforms.ToTensor())
Test_Data = datasets.CIFAR10('data',train = False,download = True,transform=transforms.ToTensor())

# Erstellen der Batches zum Trainieren und Testen
train_Batch= t.utils.data.DataLoader(Train_Data , batch_size= 64,shuffle = True)
test_Batch = t.utils.data.DataLoader(Test_Data , batch_size= 64,shuffle = False)

# Definition des CNNs für die wir die optimale Schrittweite bestimmen wollen
class Network1(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,6,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(6,12,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(8*8*12,32),
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(32,10),
            t.nn.ReLU())

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,8*8*12)
        x = self.fc1(x)
        return x


class Network2(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,8,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(8,16,3,1,1),
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

class Network3(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,12,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(12,24,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(8*8*24,32),
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(32,10))
            

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,8*8*24)

        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Network4(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,16,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(16,32,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer3 = t.nn.Sequential(
            t.nn.Conv2d(32,64,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(4*4*64,64),
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(64,10))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1,4*4*64)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
class Network5(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,16,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(16,32,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer3 = t.nn.Sequential(
            t.nn.Conv2d(32,64,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer4 = t.nn.Sequential(
            t.nn.Conv2d(64,128,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(2*2*128,32),
            t.nn.ReLU())
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
    
# Schleife zum Trainieren und Evaluieren des CNN
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

# Schrittweiten, die getestet werden sollen
Schrittweiten = [0.006,0.0055, 0.005, 0.0045, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015 , 0.001, 0.0005]
Anzahl_Gewichte = [0]

# Durchführung der Tests
Genauigkeit_CNN1 = []
for i in range(12):
    model = Network5()
    optimizer = t.optim.Adam(model.parameters(), Schrittweiten[i])
    loss_function = t.nn.CrossEntropyLoss()
    relTest,relTrain,lTest,lTrain = training_loop(5,optimizer,model,loss_function,train_Batch,test_Batch)
    Genauigkeit_CNN1.append(relTrain[-1])
print(Genauigkeit_CNN1)

Genauigkeit_CNN2 = []
for i in range(12):
    model = Network2()
    optimizer = t.optim.Adam(model.parameters(), Schrittweiten[i])
    loss_function = t.nn.CrossEntropyLoss()
    relTest,relTrain,lTest,lTrain = training_loop(5,optimizer,model,loss_function,train_Batch,test_Batch)
    Genauigkeit_CNN2.append(relTrain[-1])
print(Genauigkeit_CNN2)

Genauigkeit_CNN3 = []
for i in range(12):
    model = Network3()
    optimizer = t.optim.Adam(model.parameters(), Schrittweiten[i])
    loss_function = t.nn.CrossEntropyLoss()
    relTest,relTrain,lTest,lTrain = training_loop(5,optimizer,model,loss_function,train_Batch,test_Batch)
    Genauigkeit_CNN3.append(relTrain[-1])
print(Genauigkeit_CNN3)

Genauigkeit_CNN4 = []
for i in range(12):
    model = Network4()
    optimizer = t.optim.Adam(model.parameters(), Schrittweiten[i])
    loss_function = t.nn.CrossEntropyLoss()
    relTest,relTrain,lTest,lTrain = training_loop(5,optimizer,model,loss_function,train_Batch,test_Batch)
    Genauigkeit_CNN4.append(relTrain[-1])
print(Genauigkeit_CNN4)

Genauigkeit_CNN5 = []
for i in range(12):
    model = Network5()
    optimizer = t.optim.Adam(model.parameters(), Schrittweiten[i])
    loss_function = t.nn.CrossEntropyLoss()
    relTest,relTrain,lTest,lTrain = training_loop(5,optimizer,model,loss_function,train_Batch,test_Batch)
    Genauigkeit_CNN5.append(relTrain[-1])
print(Genauigkeit_CNN5)
    

# Visualisierung der Ergebnisse
plt.figure()
plt.plot(Schrittweiten,Genauigkeit_CNN1,color = 'blue')
plt.plot(Schrittweiten,Genauigkeit_CNN2,color = 'black')
plt.plot(Schrittweiten,Genauigkeit_CNN3,color = 'lime')
plt.plot(Schrittweiten,Genauigkeit_CNN4,color = 'magenta')
plt.plot(Schrittweiten,Genauigkeit_CNN5,color = 'red')
plt.legend([ '25766 Gewichte','34522 Gewichte','52466 Gewichte','89834 Gewichte','114186 Gewichte'],loc = 'lower center')
plt.title("Genauigkeit")
plt.ylim(0.2,0.7)
plt.savefig("Schrittweite_Netzwerkgröße.png")
plt.show()
