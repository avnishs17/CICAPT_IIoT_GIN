import torch
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from xgboost import XGBClassifier
from model.autoencoder_gin import build_model
from utils.config import build_args
from utils.utils import set_random_seed
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_metadata(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_provenance_data(data_dir='./data/provenance/processed_provenance_data'):
    train_file = os.path.join(data_dir, 'eval_data.pkl')
    metadata_file = os.path.join(data_dir, 'metadata.json')
    
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    graph, node_type_to_id, edge_type_to_id, node_attr_types, edge_attr_types = train_data
    graph = graph.to('cpu')
    
    node_attrs = []
    for attr in node_attr_types:
        if attr in graph.ndata:
            attr_tensor = graph.ndata[attr].float()
            if attr_tensor.dim() == 1:
                attr_tensor = attr_tensor.unsqueeze(1)
            node_attrs.append(attr_tensor)
    
    if node_attrs:
        graph.ndata['attr'] = torch.cat(node_attrs, dim=1)
    else:
        graph.ndata['attr'] = torch.zeros((graph.number_of_nodes(), 1))
    
    edge_attrs = []
    for attr in edge_attr_types:
        if attr in graph.edata:
            attr_tensor = graph.edata[attr].float()
            if attr_tensor.dim() == 1:
                attr_tensor = attr_tensor.unsqueeze(1)
            edge_attrs.append(attr_tensor)
    
    if edge_attrs:
        graph.edata['attr'] = torch.cat(edge_attrs, dim=1)
    else:
        graph.edata['attr'] = torch.zeros((graph.number_of_edges(), 1))
    
    return {
        'graph': graph,
        'node_type_to_id': node_type_to_id,
        'edge_type_to_id': edge_type_to_id,
        'node_attr_types': node_attr_types,
        'edge_attr_types': edge_attr_types,
        'metadata': metadata
    }

def generate_embeddings(model, graph, device):
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        embeddings = model.embed(graph)
    return embeddings.cpu().numpy()

def split_data(graph, train_ratio=0.8):
    num_nodes = graph.number_of_nodes()
    labels = graph.ndata['label'].numpy()
    
    train_indices, test_indices = train_test_split(
        np.arange(num_nodes), 
        test_size=1-train_ratio, 
        stratify=labels,
        random_state=42
    )
    
    return train_indices, test_indices

def apply_imbalance_technique(X, y, technique='smote'):
    if technique == 'smote':
        sampler = SMOTE(random_state=42)
    elif technique == 'random_under':
        sampler = RandomUnderSampler(random_state=42)
    elif technique == 'smotetomek':
        sampler = SMOTETomek(random_state=42)
    else:
        return X, y
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

def create_and_save_confusion_matrix(y_true, y_pred, dataset_name, model_name, sampling_method):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{sampling_method}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Create folder structure
    folder_path = os.path.join('confusion_matrix', dataset_name, model_name)
    os.makedirs(folder_path, exist_ok=True)
    
    # Save the confusion matrix
    image_name = f'{sampling_method}_confusion_matrix.png'
    plt.savefig(os.path.join(folder_path, image_name))
    plt.close()

def evaluate_model(clf, X_train, y_train, X_test, y_test, model_name, dataset_name, imbalance_technique=None):
    if imbalance_technique:
        X_train, y_train = apply_imbalance_technique(X_train, y_train, imbalance_technique)
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    prec, rec, threshold = precision_recall_curve(y_test, y_pred_proba)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)
    best_idx = np.argmax(f1)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    print(f'{model_name} Results ({imbalance_technique if imbalance_technique else "No resampling"}):')
    print(f'AUC: {auc:.4f}')
    print(f'Best F1: {f1[best_idx]:.4f}')
    print(f'Best Precision: {prec[best_idx]:.4f}')
    print(f'Best Recall: {rec[best_idx]:.4f}')
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    
    # Create and save confusion matrix
    sampling_method = imbalance_technique if imbalance_technique else 'No_resampling'
    create_and_save_confusion_matrix(y_test, y_pred, dataset_name, model_name, sampling_method)
    
    return auc

def main():
    args = build_args()
    set_random_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata
    metadata = load_metadata('./data/provenance/processed_provenance_data/metadata.json')
    
    # Load test data
    test_data = load_provenance_data('./data/provenance/processed_provenance_data')
    test_graph = test_data['graph']
    
    # Load model
    args.dataset = "provenance"
    args.num_hidden = 512
    args.num_layers = 6
    args.mlp_layers = 3
    args.n_dim = metadata['node_feature_dim']
    args.e_dim = metadata['edge_feature_dim']
    
    model = build_model(args)
    
    try:
        model.load_state_dict(torch.load("./checkpoints/best_model-provenance.pt", map_location=device))
    except RuntimeError as e:
        print("Error loading model state dict:", str(e))
    
    model = model.to(device)
    
    # Generate embeddings for test data
    test_embeddings = generate_embeddings(model, test_graph, device)
    
    # Split test data into train and test sets for classification
    train_indices, test_indices = split_data(test_graph)
    
    X_train = test_embeddings[train_indices]
    y_train = test_graph.ndata['label'][train_indices].numpy()
    X_test = test_embeddings[test_indices]
    y_test = test_graph.ndata['label'][test_indices].numpy()
    
    print("Data split:")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Malicious nodes in training: {np.sum(y_train == 1)}")
    print(f"Malicious nodes in testing: {np.sum(y_test == 1)}")
    
    imbalance_techniques = [None, 'smote', 'random_under', 'smotetomek']
    
    for technique in imbalance_techniques:
        print(f"\nEvaluating with {technique if technique else 'No resampling'}:")
        
        # Evaluate using SVM
        svm_clf = SVC(kernel='rbf', probability=True)
        svm_auc = evaluate_model(svm_clf, X_train, y_train, X_test, y_test, "SVM", "provenance", technique)
        
        # Evaluate using XGBoost
        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_auc = evaluate_model(xgb_clf, X_train, y_train, X_test, y_test, "XGBoost", "provenance", technique)
        
        print(f"\nFinal Results for {technique if technique else 'No resampling'}:")
        print(f"SVM AUC: {svm_auc:.4f}")
        print(f"XGBoost AUC: {xgb_auc:.4f}")

if __name__ == "__main__":
    main()