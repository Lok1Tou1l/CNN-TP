from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def preprocess_data(data_dir, batch_size):
    # Define transformations for data preprocessing
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
    ])

    # Load the dataset from the directory
    dataset = ImageFolder(root=data_dir, transform=data_transforms)

    # Calculate the sizes of train and test splits
    num_images = len(dataset)
    num_train = int(0.8 * num_images)
    num_test = num_images - num_train

    # Randomly split the dataset into train and test sets
    train_data, test_data = random_split(dataset, [num_train, num_test])

    # Create DataLoader instances for train and test sets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader
