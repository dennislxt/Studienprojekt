# Importieren der Bibliotheken
from torchvision import datasets
from torchvision import transforms
import torch as t
from matplotlib import pyplot  as plt

# Laden der Datensätze und Transformation der Bilder in das Tensor Format
Train_Data= datasets.MNIST('data',train = True, download = True, transform=transforms.ToTensor())
Test_Data = datasets.MNIST('data',train = False,download = True,transform=transforms.ToTensor())

# Erstellen der Batches zum Trainnieren und Testen des CNN
train_Batch= t.utils.data.DataLoader(Train_Data , batch_size= 32,shuffle = True)
test_Batch = t.utils.data.DataLoader(Test_Data , batch_size= 32,shuffle = False)

# Definition des CNN als Unterklasse der Klasse nn.Module aus dem torch Paket
class CNN(t.nn.Module):
    def __init__(self):
        super().__init__()
        # Erstellen von zwei Convolutional Layer
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(1,6,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(6,16,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        # Erstellen von zwei Full-Connected Layer
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(7*7*16,32),
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(32,10))
    # Definition eines Forward Passes durch das zuvor defininerte CNN
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,7*7*16)

        x = self.fc1(x)
        x = self.fc2(x)
        return x
# Funktion zum Trainieren und Testen des CNN
# Diese Funktion trainiert das CNN für epochs Epochen und berechnet die relative Genauigkeit und den Wert der Verlustfunktion für jede Epoche
def training_loop(epochs, optimizer, model,loss_function, Batch_train,Batch_test):
    # Initialisierung der Listen, in denen die Ergebnisse gespeichert werden
    correct_array_percent_test = [0]
    correct_array_percent_train = [0] 
    loss_array_train = []
    loss_array_test = []
    
    for i in range(0, epochs):
        # Schleife zum Tarinieren des CNN
        for image,label in Batch_train:
            
            outputs = model(image)
            loss = loss_function(outputs,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Schleife zum Berechnen der Genauigkeit und zur Berechnung des Werts der Verlustfunktion
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
            # Damit wir den Trainingsprozess verfolgen können, geben wir für jede Epoche die Genauigkeit aus
            print("Genauigkeit {}: {:.4f}".format(name, correct / total))
            
            # Hinzufügen der berechneten Werte zu der richtigen Liste
            if name == "train" :
                correct_array_percent_train.append(correct/total)
                loss_array_train.append(loss_value/len(train_Batch))
            if name == "test":
                correct_array_percent_test.append(correct/total)
                loss_array_test.append(loss_value/len(test_Batch))
    
    return correct_array_percent_test,correct_array_percent_train,loss_array_test,loss_array_train
# Initialisierung des CNN und Aufrufen der Funktion zum Trainieren und Testen des CNN
model = CNN()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)
loss_function = t.nn.CrossEntropyLoss()
relTest,relTrain,lTest,lTrain = training_loop(10,optimizer,model,loss_function,train_Batch,test_Batch)

# Visualisierung der Ergebnisse
x_loss = [1,2,3,4,5,6,7,8,9,10]
epochen = [0,1,2,3,4,5,6,7,8,9,10]

plt.figure()
plt.plot(x_loss,lTest,color = 'blue')
plt.plot(x_loss,lTrain,color = 'black')
plt.legend([ 'Testdaten','Trainingsdaten'],loc = 'upper right')
plt.title("Verlauf der Verlustfunktion")
plt.ylim(0,0.2)
plt.savefig("Loss_Zahlenerkennung.png")
plt.show()

plt.figure()
plt.plot(epochen,relTest,color = 'blue')
plt.plot(epochen,relTrain,color = 'black')
plt.legend([ 'Testdaten','Trainingsdaten' ],loc = 'lower right')
plt.title("Relative Genauigkeit")
plt.savefig("Rel_Zahlenerkennung.png")
plt.show()