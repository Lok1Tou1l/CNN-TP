import torch
import torch.nn as nn
import torch.optim as optim
import os
import model as net
import data_preprocessing as dp
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
        all_preds = []
        all_labels = []
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
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Testing Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    print('Training Finished!')

   
    model_save_path = os.path.join('models', 'trained_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Trained model saved to {model_save_path}')
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues_r', ax=ax);
    ax.set_ylabel('Actual label');
    ax.set_xlabel('Predicted label');
    ax.set_title('Confusion Matrix', size = 15);
    fig.savefig("data/confusion_matrix.pdf")


if __name__ == '__main__':
    data_dir = 'data'
    batch_size = 8
    num_epochs = 4
    learning_rate = 0.001
    train_model(data_dir, batch_size, num_epochs, learning_rate)
    
