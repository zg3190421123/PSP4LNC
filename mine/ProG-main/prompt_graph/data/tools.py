import torch
from torch_geometric.datasets import Planetoid, NELL
import torch_geometric.transforms as T
import numpy as np
import random
import torchmetrics
from torch_scatter import scatter_add
def load_data_new(name):
    if name == 'NELL':
        dataset = NELL(root='./' + name + '/')
        _device = torch.device('cpu')
    else:
        dataset = Planetoid(root='./data/' + name + '/', name=name,transform=T.NormalizeFeatures(), split='full')
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(_device)
    num_node_features = dataset.num_node_features

    return data, num_node_features, dataset.num_classes

def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1: # We does not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask

def refine_label_order(labels):
    print('Refine label order, Many to Few')
    num_labels = labels.max() + 1
    num_labels_each_class = np.array([(labels == i).sum().item() for i in range(num_labels)])
    sorted_index = np.argsort(num_labels_each_class)[::-1]
    idx_map = {sorted_index[i]:i for i in range(num_labels)}
    new_labels = np.vectorize(idx_map.get)(labels.numpy())

    return labels.new(new_labels), idx_map

def split_manual_lt(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels[idx_train]==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        val_lists = list(map(int,idx_val[labels[idx_val]==i]))
        test_lists = list(map(int,idx_test[labels[idx_test]==i]))
        random.shuffle(val_lists)
        random.shuffle(test_lists)

        c_num_mat[i,0] = len(c_idx)

        val_idx = val_idx + val_lists[:c_num_mat[i,1]]
        test_idx = test_idx + test_lists[:c_num_mat[i,2]]

    train_idx = torch.LongTensor(idx_train)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat