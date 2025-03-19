# PSP4LNC: Progressive Self-Supervised and Prompt-Based Learning for Long-Tail Node Classification
## Requirements

- **python version:** 3.9.21
- **numpy version:** 2.0.1
- **pytorch version:** 2.4.1  
- **torch-geometric version:** 2.6.1

## Quick Start
Here we show how to run PSP4LNC with default setting for a node classification task on Cora dataset:

```bash
python downstream_task.py --pre_train_model_path ./Experiment/pre_trained_model/Cora/Edgepred_Gprompt.GCN.128hidden_dim.pth --task NodeTask --dataset_name Cora --gnn_type GCN --prompt_type Gprompt --shot_num 1 --hid_dim 128 --num_layer 2 --lr 0.02 --decay 2e-6 --seed 42 --device 0
```


