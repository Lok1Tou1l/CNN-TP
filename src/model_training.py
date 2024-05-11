import torch
import torch.nn as nn
import torch.optim as optim
import os
import model as net
import data_preprocessing as dp

def train_model(data_dir, batch_size, num_epochs, learning_rate):
    device = torch.device('cpu')

    train_loader, test_loader = dp.preprocess_data(data_dir, batch_size)

    num_classes = len(os.listdir(data_dir))

    model = net.CNNModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')
        
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Testing Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    print('Training Finished!')

    model_save_path = os.path.join('models', 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Trained model saved to {model_save_path}')


if __name__ == '__main__':
    data_dir = 'data'
    batch_size = 8
    num_epochs = 15
    learning_rate = 0.001
    train_model(data_dir, batch_size, num_epochs, learning_rate)