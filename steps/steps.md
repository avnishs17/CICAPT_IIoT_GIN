conda env create --prefix ./env -f environment.yml

To clean up or delete the env:
conda remove --prefix ./env --all -y


# To activate this environment, use
#     $ conda activate D:\CICAPT_IIoT_GIN\env
#     $ conda activate E:\CICAPT_IIoT_GIN\env
#     $ conda activate ./env
#
# To deactivate an active environment, use
#
#     $ conda deactivate



pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html 

pip install scikit-learn==1.2.2 xgboost==1.7.5 imbalanced-learn==0.10.1 tensorboard==2.13.0


pip install matplotlib==3.7.1 seaborn==0.12.2
pip install numpy==1.24.4
pip install dgl-2.0.0+cu118-cp310-cp310-win_amd64.whl



   from utils.loaddata import load_batch_level_entity_dataset, transform_graph_entity

   phase1_csv = './data/phase1.csv'
   phase2_csv = './data/phase2.csv'
   label_column = 'label'  # Replace with your actual label column name if different

   data = load_batch_level_entity_dataset(phase1_csv, phase2_csv, label_column=label_column)


# After loading, transform the graphs using the calculated feature dimensions:
transformed_dataset = []
for graph in data['dataset']:
    transformed_graph = transform_graph_entity(graph, data['node_feat_dims'], data['edge_feat_dims'])
    transformed_dataset.append(transformed_graph)

-------------------
# You can integrate the processed data with DGL's data loaders for batching and training:

   import torch
   from torch.utils.data import DataLoader

   class DGLDatasetWrapper(torch.utils.data.Dataset):
       def __init__(self, graphs, labels):
           self.graphs = graphs
           self.labels = labels

       def __getitem__(self, idx):
           return self.graphs[idx], self.labels[idx]

       def __len__(self):
           return len(self.graphs)

   dataset_wrapper = DGLDatasetWrapper(transformed_dataset, data['labels'])
   dataloader = DataLoader(dataset_wrapper, batch_size=32, shuffle=True)




# Here's a simple example of iterating through the dataloader during training:
      for epoch in range(num_epochs):
       for batched_graph, labels in dataloader:
           # Move data to device
           batched_graph = batched_graph.to(device)
           labels = labels.to(device)
           
           # Forward pass
           outputs = model(batched_graph)
           
           # Compute loss
           loss = loss_function(outputs, labels)
           
           # Backward pass and optimization
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')