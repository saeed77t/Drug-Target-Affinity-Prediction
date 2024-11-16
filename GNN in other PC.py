import os
import torch
import pickle
import pandas as pd

def load_graph(path, is_pickle=True):
    """
    Load a molecule graph (.pkl) or a protein graph (.pt).
    If is_pickle is True, use pickle to load the file; otherwise, use torch.load.
    """
    if is_pickle:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return torch.load(path)

def prepare_dataset_individual_save_as_pt(filtered_dataset, molecule_graph_dir, protein_graph_dir, output_dir):
    """
    Incrementally prepares the dataset and saves each (molecule, protein, target) tuple as a separate .pt file.
    
    Args:
    - filtered_dataset: The filtered KIBA dataset (DataFrame).
    - molecule_graph_dir: Directory where molecule graphs are stored.
    - protein_graph_dir: Directory where protein graphs are stored.
    - output_dir: Directory to save the prepared dataset incrementally.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for index, row in filtered_dataset.iterrows():
        protein_id = row['Target_ID']
        chembl_id = row['Drug_ID']
        
        # Load the protein graph (.pt)
        pro_graph_path = os.path.join(protein_graph_dir, f"{protein_id}_graph.pt")
        if not os.path.exists(pro_graph_path):
            print(f"Protein graph not found: {protein_id}")
            continue
        pro_graph = load_graph(pro_graph_path, is_pickle=False)
        
        # Load the molecule graph (.pkl)
        mol_graph_path = os.path.join(molecule_graph_dir, f"{chembl_id}_graph.pkl")
        if not os.path.exists(mol_graph_path):
            print(f"Molecule graph not found: {chembl_id}")
            continue
        mol_graph = load_graph(mol_graph_path)

        # Load target (affinity value)
        target = torch.tensor([row['Y']], dtype=torch.float)
        
        # Create the sample as a tuple (molecule graph, protein graph, target)
        sample = (mol_graph, pro_graph, target)
        
        # Save the sample as a .pt file
        sample_path = os.path.join(output_dir, f"sample_{index}.pt")
        torch.save(sample, sample_path)

        # print(f"Saved sample {index} as {sample_path}")

# Example usage for individual saving
molecule_graph_dir = 'molecule_graphs/'  # Directory where molecule graphs are stored
protein_graph_dir = 'ProteinGraphs/'  # Directory where protein graphs are stored
filtered_dataset_path = 'filtered_KibaDataSet.csv'  # Path to the filtered dataset CSV
output_dir = 'prepared_samples/'  # Directory to save individual samples

# Load filtered dataset CSV
filtered_dataset = pd.read_csv(filtered_dataset_path)

# Prepare the dataset incrementally, saving each sample as a .pt file
prepare_dataset_individual_save_as_pt(filtered_dataset, molecule_graph_dir, protein_graph_dir, output_dir)

print("Dataset preparation completed.")




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj


# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__()

        print('GNNNet Loaded')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        # target_seq=data_pro.target

        # print('size')
        # print('mol_x', mol_x.size(), 'edge_index', mol_edge_index.size(), 'batch', mol_batch.size())
        # print('target_x', target_x.size(), 'target_edge_index', target_batch.size(), 'batch', target_batch.size())

        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        # xt = self.pro_conv4(xt, target_edge_index)
        # xt = self.relu(xt)
        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)

        # print(x.size(), xt.size())
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

    
    
    
import os
import torch
import torch.optim as optim
from torch.nn import MSELoss
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool as gep
from scipy.stats import pearsonr
import warnings
import itertools

# Suppress FutureWarning related to torch.load
warnings.filterwarnings('ignore', category=FutureWarning)


# Define the load_sample function
def load_sample(path):
    # Load individual sample from file
    sample = torch.load(path)
    mol_data = sample[0]
    pro_data = sample[1]
    target = sample[2]

    # Convert dictionaries to Data objects if necessary
    if isinstance(mol_data, dict):
        mol_data = Data(**mol_data)
    if isinstance(pro_data, dict):
        pro_data = Data(**pro_data)

    # Ensure that 'x' attribute is set
    if not hasattr(mol_data, 'x') or mol_data.x is None:
        if hasattr(mol_data, 'features'):
            mol_data.x = mol_data.features
            del mol_data.features
        else:
            raise ValueError("mol_data does not have 'x' or 'features' attribute")

    if not hasattr(pro_data, 'x') or pro_data.x is None:
        if hasattr(pro_data, 'features'):
            pro_data.x = pro_data.features
            del pro_data.features
        else:
            raise ValueError("pro_data does not have 'x' or 'features' attribute")

    # Ensure 'x' is a float tensor
    if not isinstance(mol_data.x, torch.Tensor):
        mol_data.x = torch.tensor(mol_data.x)
    if not isinstance(pro_data.x, torch.Tensor):
        pro_data.x = torch.tensor(pro_data.x)

    if mol_data.x.dtype != torch.float:
        mol_data.x = mol_data.x.float()
    if pro_data.x.dtype != torch.float:
        pro_data.x = pro_data.x.float()

    # Adjust 'edge_index' for mol_data
    # Ensure 'edge_index' is a tensor of type torch.long
    if not isinstance(mol_data.edge_index, torch.Tensor):
        mol_data.edge_index = torch.tensor(mol_data.edge_index, dtype=torch.long)
    else:
        mol_data.edge_index = mol_data.edge_index.long()

    # Ensure 'edge_index' has shape [2, num_edges]
    if mol_data.edge_index.shape[0] != 2:
        mol_data.edge_index = mol_data.edge_index.t()

    # Adjust 'edge_index' for pro_data
    if not isinstance(pro_data.edge_index, torch.Tensor):
        pro_data.edge_index = torch.tensor(pro_data.edge_index, dtype=torch.long)
    else:
        pro_data.edge_index = pro_data.edge_index.long()

    if pro_data.edge_index.shape[0] != 2:
        pro_data.edge_index = pro_data.edge_index.t()

    # Set 'num_nodes' attribute to suppress warnings
    mol_data.num_nodes = mol_data.x.size(0)
    pro_data.num_nodes = pro_data.x.size(0)

    return (mol_data, pro_data, target)

# Define the batch_loader function
def batch_loader(file_list, sample_dir, batch_size):
    batch = []
    for idx, file_name in enumerate(file_list):
        sample_path = os.path.join(sample_dir, file_name)
        sample = load_sample(sample_path)
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

# Define the evaluation metrics functions
def get_mse(y_true, y_pred):
    return np.mean((y_pred - y_true ) ** 2)

def get_ci(y_true, y_pred):
    """
    Compute the concordance index between true and predicted values.
    """
    pairs = itertools.combinations(range(len(y_true)), 2)
    c = 0
    s = 0
    for i, j in pairs:
        if y_true[i] != y_true[j]:
            s += 1
            if (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]) or \
               (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]):
                c += 1
            elif y_pred[i] == y_pred[j]:
                c += 0.5
    return c / s if s != 0 else 0

def get_pearson(y_true, y_pred):
    return pearsonr(y_true.flatten(), y_pred.flatten())[0]

def train_5fold_cross_validation(sample_dir, num_epochs=1000, n_splits=5, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}.")

    sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.pt')]
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create a single directory for all checkpoints
    training_model_dir = os.path.join(sample_dir, 'TrainingModel')
    if not os.path.exists(training_model_dir):
        os.makedirs(training_model_dir)
        print(f"Created directory for checkpoints at {training_model_dir}")
    else:
        print(f"Using existing TrainingModel directory at {training_model_dir}")

    results = []
    loss_fn = MSELoss()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(sample_files)):
        fold_number = fold + 1
        print(f'\nFold {fold_number}/{n_splits}')
        train_files = [sample_files[i] for i in train_idx]
        test_files = [sample_files[i] for i in test_idx]

        # Determine input feature dimensions from your data
        sample = load_sample(os.path.join(sample_dir, train_files[0]))
        mol_data = sample[0]
        pro_data = sample[1]

        num_features_mol = mol_data.x.size(1)
        num_features_pro = pro_data.x.size(1)

        # Initialize the GNN model with correct input dimensions
        model = GNNNet(
            num_features_mol=num_features_mol,
            num_features_pro=num_features_pro
        ).to(device)
        print(f"Model is on device: {next(model.parameters()).device}")

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Initialize starting epoch
        start_epoch = 1

        # Check for existing checkpoints in TrainingModel directory for the current fold
        existing_checkpoints = [f for f in os.listdir(training_model_dir)
                                if f.endswith('.pt') and f.startswith(f'model_fold{fold_number}_epoch')]

        if existing_checkpoints:
            # Find the latest checkpoint based on epoch number
            latest_checkpoint = max(existing_checkpoints,
                                    key=lambda x: int(x.split('_epoch')[1].split('.pt')[0]))
            checkpoint_path = os.path.join(training_model_dir, latest_checkpoint)
            print(f"Loading checkpoint for fold {fold_number} from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded_epoch = checkpoint['epoch']
            start_epoch = loaded_epoch + 1
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found for fold {fold_number}, starting training from scratch.")

        # Training loop with progress bar over epochs
        for epoch in tqdm(range(start_epoch, num_epochs + 1),
                          desc=f"Training Fold {fold_number}", unit="epoch"):
            model.train()
            running_loss = 0.0

            # Prepare batch loader without progress bar for batches
            batch_size = 256 # Adjust batch size as needed
            batch_loader_iter = batch_loader(train_files, sample_dir, batch_size=batch_size)

            for batch_samples in batch_loader_iter:
                mol_data_list = []
                pro_data_list = []
                target_list = []

                for sample in batch_samples:
                    mol_data = sample[0]
                    pro_data = sample[1]
                    target = sample[2]

                    mol_data_list.append(mol_data)
                    pro_data_list.append(pro_data)
                    target_list.append(target)

                mol_batch = Batch.from_data_list(mol_data_list).to(device)
                pro_batch = Batch.from_data_list(pro_data_list).to(device)
                target = torch.tensor(target_list, dtype=torch.float32).view(-1).to(device)

                optimizer.zero_grad()
                output = model(mol_batch, pro_batch)
                loss = loss_fn(output.view(-1), target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_samples)

            avg_loss = running_loss / len(train_files)
            # Use tqdm.write() to print without interfering with the progress bar
            tqdm.write(f"Fold {fold_number}, Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

            # Save the model and optimizer states after each epoch
            checkpoint_filename = f"model_fold{fold_number}_epoch{epoch}.pt"
            checkpoint_path = os.path.join(training_model_dir, checkpoint_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            tqdm.write(f"Checkpoint saved for fold {fold_number} at epoch {epoch}")

            # Evaluation on the test set after each epoch
            model.eval()
            total_preds, total_labels = [], []
            with torch.no_grad():
                batch_size = 256  # Adjust batch size as needed
                batch_loader_iter = batch_loader(test_files, sample_dir, batch_size=batch_size)

                for batch_samples in batch_loader_iter:
                    mol_data_list = []
                    pro_data_list = []
                    target_list = []

                    for sample in batch_samples:
                        mol_data = sample[0]
                        pro_data = sample[1]
                        target = sample[2]

                        mol_data_list.append(mol_data)
                        pro_data_list.append(pro_data)
                        target_list.append(target)

                    mol_batch = Batch.from_data_list(mol_data_list).to(device)
                    pro_batch = Batch.from_data_list(pro_data_list).to(device)
                    target = torch.tensor(target_list, dtype=torch.float32).view(-1).to(device)

                    output = model(mol_batch, pro_batch)
                    total_preds.append(output.cpu().numpy())
                    total_labels.append(target.cpu().numpy())

            # Convert lists to numpy arrays for evaluation
            total_preds = np.concatenate(total_preds)
            total_labels = np.concatenate(total_labels)

            # Calculate metrics
            mse = get_mse(total_labels, total_preds)
            ci = get_ci(total_labels, total_preds)
            pearson = get_pearson(total_labels, total_preds)

            # Print metrics
            tqdm.write(f"Fold {fold_number}, Epoch {epoch}/{num_epochs} - MSE: {mse:.4f}, CI: {ci:.4f}, Pearson: {pearson:.4f}")

        # Evaluation at the end of training for this fold
        print(f"Final evaluation for Fold {fold_number}: MSE: {mse:.4f}, CI: {ci:.4f}, Pearson: {pearson:.4f}")
        # Store results for this fold
        results.append((mse, ci, pearson))

    return results


if __name__ == "__main__":
    sample_dir = 'prepared_samples'  # Adjust the path to your samples directory
    num_epochs = 250  # Adjust the number of epochs as needed
    n_splits = 5  # Number of folds for cross-validation
    learning_rate = 0.001  # Learning rate

    # Run the training function
    results = train_5fold_cross_validation(sample_dir, num_epochs=num_epochs, n_splits=n_splits, lr=learning_rate)

    # Print overall results
    print("\nCross-validation Results:")
    for fold_idx, (mse, ci, pearson) in enumerate(results):
        print(f"Fold {fold_idx + 1}: MSE={mse:.4f}, CI={ci:.4f}, Pearson={pearson:.4f}")

    # Optionally, compute and print average metrics across folds
    mse_values, ci_values, pearson_values = zip(*results)
    print(f"\nAverage Results:")
    print(f"MSE: {np.mean(mse_values):.4f}")
    print(f"CI: {np.mean(ci_values):.4f}")
    print(f"Pearson Correlation: {np.mean(pearson_values):.4f}")
