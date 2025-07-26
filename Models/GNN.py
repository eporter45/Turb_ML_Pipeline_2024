
'''GCNconv - 6-8 hours
GraphSAGEConv 8-10 hours
GATConv 12-14 hours
GATv2Conv 14-16 hours
this is for 1000 epochs and 6 input features'''

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv
import torch_geometric.transforms as T
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import TensorDataset
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


class GNN(torch.nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            dropout,
            layer_type="GCN*",
            return_embeds=False,
            residual=False,
            num_heads=1,
            **kwargs,
    ):
        super(GNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.return_embeds = return_embeds
        self.residual = residual
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.PReLU()

        # # define layers
        # conv_layer_class = None
        # if layer_type == "GCN*":
        #     conv_layer_class = GCNConv
        # elif layer_type == "GAT":
        #     conv_layer_class = GATConv
        # elif layer_type == "GraphSAGE":
        #     conv_layer_class = SAGEConv
        # else:
        #     raise ValueError(f"Unsupported layer type \"{layer_type}\"")

        # self.convs = torch.nn.ModuleList(
        #     [conv_layer_class(input_dim, hidden_dim)]
        #     + [conv_layer_class(hidden_dim, hidden_dim) for i in range(num_layers - 1)]
        # )

        # Akoush, We cannot use the above code because each layer_type has different parameters.
        # define layers
        if layer_type == "GCN":
            self.convs = torch.nn.ModuleList(
                [GCNConv(input_dim, hidden_dim)]
                + [GCNConv(hidden_dim, hidden_dim) for i in range(num_layers - 1)]
            )
        elif layer_type == "GAT":
            self.convs = torch.nn.ModuleList(
                [GATConv(input_dim, hidden_dim, heads=num_heads)]
                + [GATConv(num_heads * hidden_dim, hidden_dim, heads=num_heads) for i in range(num_layers - 2)]
                + [GATConv(num_heads * hidden_dim, hidden_dim, heads=1)]
            )
        elif layer_type == "GATv2Conv":
            self.convs = torch.nn.ModuleList(
                [GATv2Conv(input_dim, hidden_dim, heads=num_heads)]
                + [GATv2Conv(num_heads * hidden_dim, hidden_dim, heads=num_heads) for i in range(num_layers - 2)]
                + [GATv2Conv(num_heads * hidden_dim, hidden_dim, heads=1)]
            )
        elif layer_type == "GraphSAGE":
            self.convs = torch.nn.ModuleList(
                [SAGEConv(input_dim, hidden_dim)]
                + [SAGEConv(hidden_dim, hidden_dim) for i in range(num_layers - 1)]
            )
        else:
            raise ValueError(f"Unsupported layer type \"{layer_type}\"")

        ## add ResGatedGraphConv
        self.acts = torch.nn.ModuleList(
            [torch.nn.LeakyReLU() for i in range(num_layers - 1)]
        )
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        self.linear_res = torch.nn.Linear(input_dim, hidden_dim)

    def res_block(self, conv_l, x, adj_t, first_layer=False):
        # Ensure tensors are on the correct device
        x = x.to(device)
        adj_t = adj_t.to(device)
        if first_layer:
            res = self.linear_res(x).to(device)  # Ensure residual connection is on the right device
            res = F.relu(res)
        else:
            res = x  # Skip linear projection for non-first layers
        # Apply convolution layers and ReLU/dropout
        for i in range(2):
            if first_layer and i == 1:
                x = self.convs[1](x, adj_t)  # Second conv layer for first layer
            else:
                x = conv_l(x, adj_t)  # Apply passed convolution layer
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Add residual connection
        x += res
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
        #     bn.reset_parameters()

    def forward(self, x, adj_t):
        x = x.to(device)  # Ensure x is on the correct device
        adj_t = adj_t.to(device)  # Ensure adjacency matrix is on the correct device
        for layer in range(self.num_layers - 1):
            if not self.residual:
                x = self.convs[layer](x, adj_t)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.res_block(self.convs[layer], x, adj_t, first_layer=layer == 0)

        out = self.convs[-1](x, adj_t)
        if not self.return_embeds:
            out = self.linear(out)
            #out = F.relu(out)
        return out


def train_gnn_model_with_scheduler(model, data_train, num_epochs=100, learning_rate=0.001, eval_every=10, step_size=10, gamma=0.1, batch_size=4096):
    # Loss and optimizer
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    train_loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        # Create DataLoader for batching
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

        # Total number of batches
        total_batches = len(train_loader)

        # Loop over each batch
        for batch_idx, data in enumerate(train_loader):
            # Send data to device
            data = data.to(device)
            X, adj_t, y = data.x, data.edge_index, data.y

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(X, adj_t)
            loss = criterion(outputs, y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Print the current batch number out of the total batches
            if batch_idx % (5 * eval_every) == 0:
                print(f'Batch [{batch_idx + 1}/{total_batches}] completed', end='\r')

        avg_train_loss = total_train_loss / total_batches  # Average over the number of batches
        train_loss_history.append(avg_train_loss)

        # Step the scheduler
        scheduler.step()

        if epoch % eval_every == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}]')
        # Save model every 200 to 400: save the model, optimizer, loss, and features
        # create dictionary: save for every few hundred and the smallest loss
        # for saving smallest loss,
    return train_loss_history


def run_gnn_model(data_train, data_test, num_epochs=100, learning_rate=0.001, dropout=0.1, eval_every=10, step_size=10, gamma=0.99, batch_size=4096, layer_type="GCN*", hidden_dim=64, num_layers=3, num_heads=1, residual=False):
    # Initialize the GNN model
    input_dim = data_train[0].num_features
    output_dim = data_train[0].y.shape[1]

    model = GNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        layer_type=layer_type,
        num_heads=num_heads,
        residual=residual
    ).to(device)

    # Train the model using the previously defined function
    train_loss_history = train_gnn_model_with_scheduler(
        model,
        data_train,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        eval_every=eval_every,
        step_size=step_size,
        gamma=gamma,
        batch_size=batch_size
    )

    # Make predictions on the test data
    predictions = make_gnn_predictions(model, data_test)

    return predictions, train_loss_history, model


def train_gnn_model_with_scheduler2(model, data_train, data_test, directory, zweight, num_epochs=100,
                                    learning_rate=0.001, eval_every=10, step_size=10, gamma=0.1, batch_size=4096):
    # Loss and optimizer
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # For calculating losses
    loss_weights = [1, 1, zweight, 1, zweight, 1]
    weights = torch.tensor(loss_weights, dtype=torch.float, device=device)

    # History tracking
    train_loss_history = []

    # Best model tracking
    smallest_loss = float('inf')
    best_loss_epoch = 0

    # Early stopping setup
    best_test_loss = float('inf')
    early_stop = False
    no_improvement_epochs = 0  # Counter for early stopping
    patience = 250  # Number of epochs to wait before stopping
    ranges = [((0, 400), 150), ((400, 1000), 300), ((1000, num_epochs), 500)]

    for epoch in range(num_epochs):

        if early_stop:
            print(f'Early stopping at epoch {epoch}')
            break

        model.train()
        total_train_loss = 0
        # Create DataLoader for batching
        train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        # Total number of batches
        total_batches = len(train_loader)
        # Loop over each batch
        for batch_idx, data in enumerate(train_loader):
            # Send data to device
            data = data.to(device)
            X, adj_t, y = data.x, data.edge_index, data.y
            # Move the tensors to the appropriate device (e.g., GPU if available)
            X = X.to(device)
            adj_t = adj_t.to(device)
            y = y.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X, adj_t)
            # Calculate loss and apply weights
            loss = criterion(outputs, y)
            weighted_loss = loss * weights
            final_weighted_loss = weighted_loss.mean()
            # Backward pass and optimize
            final_weighted_loss.backward()
            optimizer.step()
            total_train_loss += final_weighted_loss.item()
            # Print progress for every few batches
            if batch_idx % (5 * eval_every) == 0:
                print(f'Batch [{batch_idx + 1}/{total_batches}] completed', end='\r')
        # Average training loss for this epoch
        avg_train_loss = total_train_loss / total_batches
        train_loss_history.append(avg_train_loss)
        # Step the scheduler
        scheduler.step()
        # Print training loss
        if epoch % eval_every == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}')

        # Save model, model state dict, and optimizer every 250 epochs
        if epoch % 250 == 0 and epoch != 0:
            epoch_dir = os.path.join(directory, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            # Define file paths
            model_save_path = os.path.join(epoch_dir, 'model.pth')
            model_dict_save_path = os.path.join(epoch_dir, 'model_dict.pth')
            optimizer_dict_save_path = os.path.join(epoch_dir, 'optimizer_dict.pth')
            # Save model, state dict, and optimizer state
            torch.save(model, model_save_path)
            torch.save(model.state_dict(), model_dict_save_path)
            torch.save(optimizer.state_dict(), optimizer_dict_save_path)
            print(f'Model, model state dict, and optimizer saved at epoch {epoch} in {epoch_dir}')
        # Check smallest losses, and save model every 300 epochs and if it is the best loss
        if epoch > 300 and epoch != 0 and epoch % 20 == 0:
            if avg_train_loss < smallest_loss:
                best_loss_epoch = epoch
                smallest_loss = avg_train_loss
                print(f'Smallest loss updated on epoch {epoch}')

                epoch_dir = os.path.join(directory, 'epoch_best_loss')
                os.makedirs(epoch_dir, exist_ok=True)

                # Save model, state dict, and optimizer for the best loss
                model_save_path = os.path.join(epoch_dir, 'model.pth')
                model_dict_save_path = os.path.join(epoch_dir, 'model_dict.pth')
                optimizer_dict_save_path = os.path.join(epoch_dir, 'optimizer_dict.pth')

                torch.save(model, model_save_path)
                torch.save(model.state_dict(), model_dict_save_path)
                torch.save(optimizer.state_dict(), optimizer_dict_save_path)

                print(f'Model with the lowest loss saved at epoch {epoch} in {epoch_dir}')

    return train_loss_history, best_loss_epoch, optimizer, model.state_dict()


def run_gnn_model2(data_train, data_test, directory, z_weights, num_epochs=100, learning_rate=0.001, dropout=0.1, eval_every=10, step_size=10, gamma=0.99, batch_size=4096, layer_type="GCN", hidden_dim=64, num_layers=3, num_heads=1, residual=False):
    # Initialize the GNN model
    input_dim = data_train[0].num_features
    output_dim = data_train[0].y.shape[1]

    model = GNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        layer_type=layer_type,
        num_heads=num_heads,
        residual=residual
    ).to(device)

    # Train the model using the previously defined function
    train_loss_history, best_epoch, optimizer, model_state_dict = train_gnn_model_with_scheduler2(
        model,
        data_train= data_train,
        data_test= data_test,
        num_epochs=num_epochs,
        directory=directory,
        zweight= z_weights,
        learning_rate=learning_rate,
        eval_every=eval_every,
        step_size=step_size,
        gamma=gamma,
        batch_size=batch_size)

    # Make predictions on the test data
    predictions = make_gnn_predictions(model, data_test)

    return predictions, train_loss_history, model, best_epoch, optimizer, model_state_dict


def make_gnn_predictions(model, data_test):
    model.eval()  # Set model to evaluation mode (turn off dropout, batch norm, etc.)
    outputs = []
    with torch.no_grad():  # Disable gradient calculation for prediction
        for data in data_test:
            # Ensure data is on the correct device
            data = data.to(device)
            # Forward pass: make predictions
            pred = model(data.x.to(device), data.edge_index.to(device))
            # Move prediction to CPU before appending
            outputs.append(pred.cpu())

    return outputs


def compute_test_losses_gnn(model, test_data_list, criterion, weights, batch_size=4096):
    """
    Helper function to calculate test losses over GNN test data.

    Args:
    - model: The trained GNN model.
    - test_data_list: List of test data graphs (e.g., PyG Data objects).
    - criterion: The loss function (e.g., nn.MSELoss).
    - weights: The weight tensor to apply for specific outputs.
    - batch_size: The batch size to use for testing.

    Returns:
    - total_test_loss: The total loss over all test batches.
    - avg_test_loss: The average loss across all batches.
    """
    model.eval()  # Set model to evaluation mode (turn off dropout, batch norm, etc.)
    total_test_loss = 0
    test_loader = DataLoader(test_data_list, batch_size=batch_size, shuffle=False)

    with torch.no_grad():  # Disable gradient calculation in evaluation
        for batch_data in test_loader:
            # Move batch data to the appropriate device (GPU or CPU)
            batch_data = batch_data.to(device)
            # Extract node features, adjacency matrix (edge index), and labels (y)
            x = batch_data.x.to(device)
            adj_t = batch_data.edge_index.to(device)
            y = batch_data.y.to(device)
            # Forward pass to get model outputs
            outputs = model(x, adj_t)
            # Calculate the loss
            loss = criterion(outputs, y)
            # Apply the weights to the loss (element-wise multiplication)
            weighted_loss = loss * weights.to(device)
            final_weighted_loss = weighted_loss.mean()
            # No need to perform backward pass in testing
            total_test_loss += final_weighted_loss.item()
    # Calculate average loss over all batches
    avg_test_loss = total_test_loss / len(test_loader)

    return total_test_loss, avg_test_loss
