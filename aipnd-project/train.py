import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import argparse
from model_definition import Classifier

def define_default_data(train_dir_set, custom_data_dir):
    if train_dir_set:
        data_dir = 'flowers'
        train_dir = custom_data_dir
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
    else:
        data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    return trainloader, validloader, testloader, train_data

def declare_model(arch, learning_rate, hidden_units):
    if arch == "vgg":
        model = models.VGG(pretrained=True)
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        print("improper arch definded (please choose vgg. vgg11, vgg13, or vgg19(default))")
        return
    
    #declare model, optimizer, and criterion
    model.classifier = Classifier(hidden_units)
    # Define the loss function
    criterion = nn.NLLLoss()
    
    # Set the optimizer and learning rate
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, criterion, optimizer

def train_model(model, optimizer, criterion, epochs, gpu, trainloader, testloader ): 
    
    running_loss = 0
    device = "cpu"
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            # Switch to the train mode
            model.train()
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
        #now do some validation while training to see learning progress
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
    
                test_loss += batch_loss.item() 
    
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss:.3f}.. "
              f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")
        running_loss = 0
    return model, optimizer

def save_ceckpoint(model, optimizer, train_data, arch, epochs, save_dir):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'model' : arch,
                  'classifier' : model.classifier,
                  'epochs' : epochs,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx }
    save_path = save_dir+'checkpoint.pth'
    print("checkpoint saved at: ", save_path)
    torch.save(checkpoint, save_path)


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--data_dir', help='path to data folder', required=False)
    parser.add_argument('--save_dir', help='dir where checkpoint should be saved', required=False)
    parser.add_argument('--arch', help='architechture to be used (vgg19 default)', required=False)
    parser.add_argument('--epochs', help='# of epochs', required=False)
    parser.add_argument('--learning_rate', help='learning rate of NN', required=False)
    parser.add_argument('--hidden_units', help='hidden units in NN', required=False)
    parser.add_argument('--gpu', help='train with gpu', action='store_true')
    args = vars(parser.parse_args())
    
    #defaults 
    data_dir = './flowers/valid'
    train_dir_set = False
    save_dir = './'
    arch = 'vgg19'
    epochs = 5
    learning_rate = 0.003
    hidden_units = 1568
    gpu = False
    
    #get args
    if args['data_dir']:
        data_dir = args['data_dir']
        train_dir_set = True
        data_dir = data_dir.replace('\\','/') #for windows paths
        if data_dir[len(data_dir)-1] != '/':
            data_dir = data_dir + "/"
    
    if args['save_dir']:
        save_dir = args['save_dir']
        save_dir = save_dir.replace('\\','/') #for windows paths
        if save_dir[len(save_dir)-1] != '/':
            save_dir = save_dir + "/"
        
    if args['arch']:
        arch = args['arch']
    if args['epochs']:
        epochs = int(args['epochs'])
    if args['learning_rate']:
        learning_rate = float(args['learning_rate'])
    if args['hidden_units']:
        hidden_units = int(args['hidden_units'])
    if args['gpu']:
        gpu = True
    
    #define trainloaders
    trainloader, validloader, testloader, train_data = define_default_data(train_dir_set, data_dir)
    #define model
    model, criterion, optimizer = declare_model(arch, learning_rate, hidden_units)
    #train model
    print("training model, please wait...")
    model, optimizer = train_model(model, optimizer, criterion, epochs, gpu, trainloader, testloader )
    save_ceckpoint(model, optimizer, train_data, arch, epochs, save_dir)
    print("Done")
    
    
    
main()