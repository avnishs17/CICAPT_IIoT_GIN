import os
import torch
import warnings
from tqdm import tqdm
import dgl
from utils.prov_load import load_provenance_data
from model.autoencoder_gin import build_model
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
from torch.utils.tensorboard import SummaryWriter
import time

warnings.filterwarnings('ignore')

def get_unique_log_dir(base_dir):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_dir, "provenance", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def split_graph(graph, val_ratio=0.1):
    num_nodes = graph.number_of_nodes()
    num_val = int(num_nodes * val_ratio)
    
    perm = torch.randperm(num_nodes)
    val_indices = perm[:num_val]
    train_indices = perm[num_val:]
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    
    return train_mask, val_mask

def train(model, graph, optimizer, device, max_epoch, patience=10):
    log_dir = get_unique_log_dir("runs")
    writer = SummaryWriter(log_dir)
    epoch_iter = tqdm(range(max_epoch))
    
    graph = graph.to(device)
    train_mask, val_mask = split_graph(graph)
    graph.ndata['train_mask'] = train_mask.to(device)
    graph.ndata['val_mask'] = val_mask.to(device)
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()
        loss = model(graph)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss = loss.item()
        
        model.eval()
        with torch.no_grad():
            val_loss = model(graph).item()
        

        writer.add_scalars('Loss', {
                'Train': train_loss,
                'Validation': val_loss
            }, epoch)
                               
        epoch_iter.set_description(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    writer.close()
    return best_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(0)

    # Load provenance data
    data = load_provenance_data()
    graph = data['graph']

    # Set dimensions based on the actual data
    args.n_dim = graph.ndata['attr'].shape[1]
    args.e_dim = graph.edata['attr'].shape[1]

    print(f"Node feature dimension: {args.n_dim}")
    print(f"Edge feature dimension: {args.e_dim}")

    # Build and initialize the model
    model = build_model(args)
    model = model.to(device)

    # Create optimizer
    optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    # Train the model
    best_model = train(model, graph, optimizer, device, args.max_epoch)

    # Save the best model
    torch.save(best_model, "./checkpoints/best_model-provenance.pt")

    print("Training completed. Best model saved.")

if __name__ == '__main__':
    args = build_args()
    args.dataset = "provenance"
    args.num_hidden = 512
    args.max_epoch = 50
    args.num_layers = 6
    args.mlp_layers = 3
    main(args)