from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot  as plt
import torch as t

# Laden der Daten mit der ImageFolder Funktion
Train_Data = datasets.ImageFolder(root='root Eintragen',transform=transforms.Compose([transforms.Resize(size = (30,45)) , transforms.ToTensor()]))
Test_Data = datasets.ImageFolder(root='root Eintragen',transform=transforms.Compose([transforms.Resize(size = (30,45)) , transforms.ToTensor()]))

# Erstellen der Batches
Train_Batch = t.utils.data.DataLoader(Train_Data , batch_size= 12,shuffle = True)
Test_Batch = t.utils.data.DataLoader(Test_Data , batch_size= 12,shuffle = False)

# Definition des CNN
class CNN(t.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,32,3,1,1),
            t.nn.BatchNorm2d(32),
            t.nn.ReLU(), t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(32,64,3,1,1),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(), t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Dropout(p=0.2),
            t.nn.Linear(7*11*64,64),
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Dropout(p=0.2),
            t.nn.Linear(64,3))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 7 * 11 * 64)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Funktion zum Trainieren und Testen des CNN
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
                loss_array_train.append(loss_value/len(Train_Batch))
            if name == "test":
                correct_array_percent_test.append(correct/total)
                loss_array_test.append(loss_value/len(Test_Batch))
    
    return correct_array_percent_test,correct_array_percent_train,loss_array_test,loss_array_train

# Initialisierung und Durchführung des Training und Test
model = CNN()
optimizer = t.optim.Adam(model.parameters(), lr=0.0005)
loss_function = t.nn.CrossEntropyLoss()
relTest,relTrain,lTest,lTrain = training_loop(18,optimizer,model,loss_function,Train_Batch,Test_Batch)

# Visualsisierung der Ergebnisse
x_loss = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
epochen = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

plt.figure()
plt.plot(x_loss,lTest,color = 'magenta')
plt.plot(x_loss,lTrain,color = 'black')
plt.legend([ 'Testdaten','Trainingsdaten'],loc = 'upper right')
plt.title("Verlauf der Verlustfunktion")
plt.savefig("Loss_Objekterkennung.png")
plt.show()

plt.figure()
plt.plot(epochen,relTest,color = 'magenta')
plt.plot(epochen,relTrain,color = 'black')
plt.legend([ 'Testdaten','Trainingsdaten' ],loc = 'lower right')
plt.ylim(0,1)
plt.title("Relative Genauigkeit")
plt.savefig("Rel_Objekterkennung.png")
plt.show()