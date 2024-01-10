import argparse
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch

def main():
    user_input = get_input_args()
    arch = in_arg.arch
    data_dir = in_arg.data_directory
    epoch = in_arg.epoch
    gpu = in_arg.gpu
    hidden_units = in_arg.hidden_units
    rate = in_arg.rate
    save_dir = in_arg.save_dir


    size = 64
    epoch = 10
    rate = 0.001
    print_time = 20
    data_dir = 'flowers'
    training_dir = data_dir + '/train'
    validation_dir = data_dir + '/valid'
    testing_dir = data_dir + '/test'
    input_size = 1024
    output_size = 102

    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    
    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(training_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(validation_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(testing_dir ,transform = testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(training_data, batch_size=size, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =size,shuffle = True)
    testloader = torch.utils.data.DataLoader(testing_data, batch_size = size, shuffle = True)

    model = models.densenet121(pretrained = True)
    from collections import OrderedDict
    for p in model.parameters():
        p.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, 512)),('relu1', nn.ReLU()),('dropout1', nn.Dropout(p = 0.5)),('fc2', nn.Linear(512, 256)),('relu2', nn.ReLU()),('fc3', nn.Linear(256, 102)),('output', nn.LogSoftmax(dim = 1))]))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = rate )

    training_model(model, trainloader, testloader, epoch, print_every, criterion, optimizer)

    checkpoint = {
    'epochs': epoch,
    'input_size': input_size,
    'output_size': output_size,
    'rate': rate,
    'size': size,
    'validation_transforms': validation_transforms,
    'model': models.densenet121(pretrained=True),
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': model.class_to_idx,
    'classifier': model.classifier,
}
    torch.save(checkpoint, 'checkpoint.pth')

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory',action = 'store',)
    parser.add_argument('--save_dir',type = str,dest = 'save_dir',default = '',)
    parser.add_argument('--arch',type = str,dest = 'arch',default = 'densenet121')
    parser.add_argument('--rate',type = float,dest = 'rate',default = 0.001,)
    parser.add_argument('--hidden_units',type = int,dest = 'hidden_units',default = 1024,)
    parser.add_argument('--epoch',type = int,dest = 'epoch',default = 10,)
    parser.add_argument('--gpu',dest = 'gpu',action = 'store_true',)
    return parser.parse_args()

def training_model(model, trainloader, testloader, epoch, print_time, criterion, optimizer):
    steps = 0
    input_size = 1024
    output_size = 102
    model = model.to(device)
    model.train()
    print("Training started")
    for e in range(epoch):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_time == 0:
                model.eval()
                accuracy = 0
                validation_loss  = 0
                for ii, (inputs, labels) in enumerate(testloader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    validation_loss  += criterion(output, labels)
                    probabilities = torch.exp(output).data
                    equality = (labels.data == probabilities.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                print("Epoch: {}/{}... ".format(e+1, epoch),
                      "| Training Loss: {:.4f}".format(running_loss / print_time),
                      "| Validation Loss: {:.3f}.. ".format(validation_loss  / len(testloader)),
                      "| Validation Accuracy: {:.3f}%".format(accuracy / len(testloader) * 100))
                running_loss = 0
                model.train()
    
    print("Done!")

if __name__ == '__main__':
    main()
