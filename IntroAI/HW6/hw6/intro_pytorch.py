import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    test_set = datasets.FashionMNIST('./data', train=False, download=False, transform=custom_transform)

    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
    test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    data_loader = train_data_loader if training else test_data_loader

    return data_loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10, bias=True)
        )

    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    model.train()

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):  # loop over the dataset multiple times

        running_loss = 0.0
        all_samples = 0
        correct_samples = 0
        batch_num = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_samples += labels.size(0)
            correct_samples += (predicted == labels).sum().item()
            batch_num += 1

        print("Train Epoch: {} Accuracy: {}/{}({:.2f}%) Loss: {:.3f}".format(epoch, correct_samples, all_samples, 100*correct_samples/all_samples, running_loss/batch_num))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()

    running_loss = 0.0
    all_samples = 0
    correct_samples = 0
    batch_num = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            running_loss += criterion(outputs, labels).item()
            all_samples += outputs.size()[0]
            _, predicted = torch.max(outputs.data, 1)
            correct_samples += (predicted == labels).sum().item()
            batch_num += 1

    if show_loss:
        print("Average loss: {:.4f}".format(running_loss/batch_num))

    print("Accuracy: {:.2f}%".format(100 * correct_samples / all_samples))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """

    test_image = test_images[index].float()
    output = model(test_image)
    prob = F.softmax(output, dim=1)
    values, indices = torch.sort(prob, descending=True)
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
    for i in range(3):
        print("{}: {:.2f}%".format(class_names[indices[0][i].item()], 100*values[0][i].item()))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    train_loader = get_data_loader(True)
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, False)
    test_data = test_loader.dataset.data
    n = test_data.size()[0]
    m = test_data.size()[1]
    predict_label(model, test_data.reshape(n, 1, m, m).float(), 0)
