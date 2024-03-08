from sklearn.calibration import LabelEncoder
import torch
import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    A dataset class for handling CSV files that can process categorical data and apply transformations.

    Attributes:
        data (DataFrame): The dataset loaded from a CSV file.
        transform (callable, optional): Optional transform to be applied on a sample.
        label_encoders (dict of LabelEncoder): Encoders for transforming categorical columns to numerical data.
        categorical_column_names (list of str): List of column names in the dataset that are categorical.
    """

    def __init__(self, file_path, categorical_column_names=None, transform=None):
        
        """
        Initializes the CSVDataset object by loading data, encoding categorical columns, and setting transformations.

        Args:
            file_path (str): Path to the CSV file.
            categorical_column_names (list of str): List of the names of the categorical columns.
            transform (callable, optional): Optional transform to be applied on a sample.
        """        
        self.data = pd.read_csv(file_path,header=None)
        if categorical_column_names is None:
            categorical_column_names = []

        
        
        self.transform = transform
        self.label_encoders = {}
        self.categorical_column_names = categorical_column_names
        
        # Encode the categorical columns
        for column_name in self.categorical_column_names:
            self.label_encoders[column_name] = LabelEncoder()
            categorical_column = self.data[column_name].astype(float)
            encoded_categorical_column = self.label_encoders[column_name].fit_transform(categorical_column)
            self.data[column_name] = encoded_categorical_column
    
    def __len__(self):
        """
        Returns the number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """        
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Retrieve an item by its index and optionally apply a transformation.

        Args:
            index (int): Index of the desired item.

        Returns:
            Tensor: Transformed data item as a tensor.
        """
        item = self.data.iloc[index]
        
        # Apply data transformation if provided
        if self.transform:
            item = self.transform(item)
        
        # Convert item to tensors if needed
        item = torch.tensor(item).float()
        
        return item
    
    def calculate_feature_means(self):
        """
        Calculates and returns the mean of each feature in the dataset.

        Returns:
            list[float]: A list containing the mean of each feature.
        """        
        category_means = []
        
        # Calculate means for each category
        for column_name in self.data.columns:
            category_means.append(self.data[[column_name]].mean().iloc[0])
            
        return category_means
 
class Logit:
    """Transform for dataloader

    ```
    logit(alpha + (1 - alpha) * x) where logit(x) = log(x / (1 - x))
    ```
    """

    def __init__(self, alpha=0):
        """Constructor

        Args:
          alpha: see above
        """
        self.alpha = alpha

    def __call__(self, x):
        x_ = self.alpha + (1 - self.alpha) * x
        return torch.log(x_ / (1 - x_))

    def inverse(self, x):
        return (torch.sigmoid(x) - self.alpha) / (1 - self.alpha)


class Jitter:
    """Transform for dataloader, adds uniform jitter noise to data"""

    def __init__(self, scale=1.0 / 256):
        """Constructor

        Args:
          scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        eps = torch.rand_like(x) * self.scale
        x_ = x + eps
        return x_


class Scale:
    """Transform for dataloader, adds uniform jitter noise to data"""

    def __init__(self, scale=255.0 / 256.0):
        """Constructor

        Args:
          scale: Scaling factor for noise
        """
        self.scale = scale

    def __call__(self, x):
        return x * self.scale
