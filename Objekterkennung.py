from torchvision import datasets
from torchvision import transforms
import torch as t
from matplotlib import pyplot  as plt

# Alles analog zur Zahlenerkennung, bis auf die markierten Anpassungen

# Laden des CIFAR10 Datensatzes statt des MNIST Datensatzes
Train_Data= datasets.CIFAR10('data',train = True, download = True, transform=transforms.ToTensor())
Test_Data = datasets.CIFAR10('data',train = False,download = True,transform=transforms.ToTensor())

train_Batch= t.utils.data.DataLoader(Train_Data , batch_size= 32,shuffle = True)
test_Batch = t.utils.data.DataLoader(Test_Data , batch_size= 32,shuffle = False)

class CNN(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = t.nn.Sequential(
            t.nn.Conv2d(3,6,3,1,1), # Anpassung der Inputschichten, da nun Farbbilder verwendet werden
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.layer2 = t.nn.Sequential(
            t.nn.Conv2d(6,16,3,1,1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(2))
        self.fc1 = t.nn.Sequential(
            t.nn.Linear(8*8*16,32), # Anpassung der Größe des FC-Layer, da CIFAR10 Bilder 4 Pixel Größer sind als MNIST
            t.nn.ReLU())
        self.fc2 = t.nn.Sequential(
            t.nn.Linear(32,10))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,8*8*16) # Anpassung der Größe, da CIFAR10 Bilder 4 Pixel Größer sind als MNIST

        x = self.fc1(x)
        x = self.fc2(x)
        return x

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
        
model = CNN()
optimizer = t.optim.Adam(model.parameters(), lr=0.01)
loss_function = t.nn.CrossEntropyLoss()
relTest,relTrain,lTest,lTrain = training_loop(10,optimizer,model,loss_function,train_Batch,test_Batch)

x_loss = [1,2,3,4,5,6,7,8,9,10]
epochen = [0,1,2,3,4,5,6,7,8,9,10]

plt.figure()
plt.plot(x_loss,lTest,color = 'blue')
plt.plot(x_loss,lTrain,color = 'black')
plt.legend([ 'Testdaten','Trainingsdaten'],loc = 'upper right')
plt.title("Verlauf der Verlustfunktion")
plt.ylim(0,1.9)
plt.savefig("Loss_Objekterkennung.png")
plt.show()

plt.figure()
plt.plot(epochen,relTest,color = 'blue')
plt.plot(epochen,relTrain,color = 'black')
plt.legend([ 'Testdaten','Trainingsdaten' ],loc = 'lower right')
plt.ylim(0,1)
plt.title("Relative Genauigkeit")
plt.savefig("Rel_Objekterkennung.png")
plt.show()