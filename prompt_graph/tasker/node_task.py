import torch
from torch_geometric.loader import DataLoader
from prompt_graph.utils import constraint,  center_embedding, Gprompt_tuning_loss
from prompt_graph.evaluation import GPPTEva, GNNNodeEva, GPFEva, MultiGpromptEva
from prompt_graph.pretrain import GraphPrePrompt, NodePrePrompt, prompt_pretrain_sample
from .task import BaseTask
from prompt_graph.utils import seed_everything
import time
import warnings
import numpy as np
from prompt_graph.data import load4node, induced_graphs, graph_split, split_induced_graphs, node_sample_and_save,GraphDataset
from prompt_graph.evaluation import GpromptEva, GpromptEva_new,AllInOneEva
import pickle
import os
import random
from prompt_graph.utils import process
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.optim as optim
warnings.filterwarnings("ignore")

class NodeTask(BaseTask):
      def __init__(self, data, input_dim, output_dim, graphs_list = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_type = 'NodeTask'
            if self.prompt_type == 'MultiGprompt':
                  self.load_multigprompt_data()
            else:
                  self.data = data
                  if self.dataset_name == 'ogbn-arxiv':
                        self.data.y = self.data.y.squeeze()
                  self.input_dim = input_dim
                  self.output_dim = output_dim
                  self.graphs_list = graphs_list
                  self.answering =  torch.nn.Sequential(torch.nn.Linear(self.hid_dim, self.output_dim),
                                                torch.nn.Softmax(dim=1)).to(self.device) 
            
            self.create_few_data_folder()

      def create_few_data_folder(self):
      # 创建文件夹并保存数据
            batch_folder = './Experiment/sample_data/Node/'+ self.dataset_name +'/' +'batch'
            os.makedirs(batch_folder, exist_ok=True)

            for i in range(1, 11):
                  folder = os.path.join(batch_folder, str(i))
                  if not os.path.exists(folder):
                        os.makedirs(folder)
                        node_sample_and_save(self.data, folder, self.output_dim)
                        print(' batch ' + str(i) + ' th is saved!!')

      def load_multigprompt_data(self):
            adj, features, labels = process.load_data(self.dataset_name)
            # adj, features, labels = process.load_data(self.dataset_name)  
            self.input_dim = features.shape[1]
            self.output_dim = labels.shape[1]
            print('a',self.output_dim)
            features, _ = process.preprocess_features(features)
            self.sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            self.labels = torch.FloatTensor(labels[np.newaxis])
            self.features = torch.FloatTensor(features[np.newaxis]).to(self.device)
            # print("labels",labels)
            print("adj",self.sp_adj.shape)
            print("feature",features.shape)

      def load_data(self):
            self.data, self.input_dim, self.output_dim = load4node(self.dataset_name)

      def train(self, data, train_idx):
            self.gnn.train()
            self.answering.train()
            self.optimizer.zero_grad() 
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss.backward()  
            self.optimizer.step()  
            return loss.item()
      
      def GPPTtrain(self, data, train_idx):
            self.prompt.train()
            node_embedding = self.gnn(data.x, data.edge_index)
            out = self.prompt(node_embedding, data.edge_index)
            loss = self.criterion(out[train_idx], data.y[train_idx])
            loss = loss + 0.001 * constraint(self.device, self.prompt.get_TaskToken())
            self.pg_opi.zero_grad()
            loss.backward()
            self.pg_opi.step()
            mid_h = self.prompt.get_mid_h()
            self.prompt.update_StructureToken_weight(mid_h)
            return loss.item()
      
      def MultiGpromptTrain(self, pretrain_embs, train_lbls, train_idx):
            self.DownPrompt.train()
            self.optimizer.zero_grad()
            prompt_feature = self.feature_prompt(self.features)
            # prompt_feature = self.feature_prompt(self.data.x)
            # embeds1 = self.gnn(prompt_feature, self.data.edge_index)
            embeds1= self.Preprompt.gcn(prompt_feature, self.sp_adj , True, False)
            pretrain_embs1 = embeds1[0, train_idx]
            logits = self.DownPrompt(pretrain_embs,pretrain_embs1, train_lbls,1).float().to(self.device)
            loss = self.criterion(logits, train_lbls)           
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return loss.item()
      
      def SUPTtrain(self, data):
            self.gnn.train()
            self.optimizer.zero_grad() 
            data.x = self.prompt.add(data.x)
            out = self.gnn(data.x, data.edge_index, batch=None) 
            out = self.answering(out)
            loss = self.criterion(out[data.train_mask], data.y[data.train_mask])  
            orth_loss = self.prompt.orthogonal_loss()
            loss += orth_loss
            loss.backward()  
            self.optimizer.step()  
            return loss
      
      def GPFTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            for batch in train_loader:  
                  self.optimizer.zero_grad() 
                  batch = batch.to(self.device)
                  batch.x = self.prompt.add(batch.x)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = self.prompt_type)
                  out = self.answering(out)
                  loss = self.criterion(out, batch.y)  
                  loss.backward()  
                  self.optimizer.step()  
                  total_loss += loss.item()  
            return total_loss / len(train_loader) 

      def AllInOneTrain(self, train_loader, answer_epoch=1, prompt_epoch=1):
            #we update answering and prompt alternately.
            # tune task head
            self.answering.train()
            self.prompt.eval()
            self.gnn.eval()
            for epoch in range(1, answer_epoch + 1):
                  answer_loss = self.prompt.Tune(train_loader, self.gnn,  self.answering, self.criterion, self.answer_opi, self.device)
                  print(("frozen gnn | frozen prompt | *tune answering function... {}/{} ,loss: {:.4f} ".format(epoch, answer_epoch, answer_loss)))

            # tune prompt
            self.answering.eval()
            self.prompt.train()
            for epoch in range(1, prompt_epoch + 1):
                  pg_loss = self.prompt.Tune( train_loader,  self.gnn, self.answering, self.criterion, self.pg_opi, self.device)
                  print(("frozen gnn | *tune prompt |frozen answering function... {}/{} ,loss: {:.4f} ".format(epoch, prompt_epoch, pg_loss)))
            
            # return pg_loss
            return answer_loss
      
      def GpromptTrain(self, train_loader):
            self.prompt.train()
            total_loss = 0.0 
            accumulated_centers = None
            accumulated_counts = None
            for batch in train_loader:  
                  self.pg_opi.zero_grad() 
                  batch = batch.to(self.device)
                  out = self.gnn(batch.x, batch.edge_index, batch.batch, prompt = self.prompt, prompt_type = 'Gprompt')
                  # out = s𝑡,𝑥 = ReadOut({p𝑡 ⊙ h𝑣 : 𝑣 ∈ 𝑉 (𝑆𝑥)}),
                  center, class_counts = center_embedding(out, batch.y, self.output_dim)
                   # 累积中心向量和样本数
                  if accumulated_centers is None:
                        accumulated_centers = center
                        accumulated_counts = class_counts
                  else:
                        accumulated_centers += center * class_counts
                        accumulated_counts += class_counts
                  criterion = Gprompt_tuning_loss()
                  loss = criterion(out, center, batch.y)  
                  loss.backward()  
                  self.pg_opi.step()  
                  total_loss += loss.item()

            # 处理 accumulated_counts 中的零值
            safe_accumulated_counts = accumulated_counts.clone()
            safe_accumulated_counts[safe_accumulated_counts == 0] = 1

            # 计算 mean_center，避免出现 NaN 值
            mean_centers = accumulated_centers / safe_accumulated_counts

            # 将原来 accumulated_counts 为零的地方重置为零
            # mean_centers[accumulated_counts == 0] = 0
            # 计算加权平均中心向量
            # mean_centers = accumulated_centers / accumulated_counts

            return total_loss / len(train_loader), mean_centers
      
      def run(self):
            test_accs = []
            f1s = []
            rocs = []
            prcs = []
            #以Cora数据集为例
            head_label = [2,3]
            body_label = [0,4]
            tail_label = [1,5,6]
            centers = {}

            if self.prompt_type == 'All-in-one':
                  self.answer_epoch = 50
                  self.prompt_epoch = 50
                  self.epochs = int(self.epochs/self.answer_epoch)

            for i in range(1, 2):
                  seed_everything(42+i)
                  batch_idx = i
                  self.initialize_gnn()
                  self.initialize_prompt()
                  self.initialize_optimizer()
                  idx_train = torch.load("./Experiment/sample_data/Node/{}/batch/{}/train_idx.pt".format(self.dataset_name, i)).type(torch.long).to(self.device)
                  # print('idx_train',idx_train)
                  train_lbls = torch.load("./Experiment/sample_data/Node/{}/batch/{}/train_labels.pt".format(self.dataset_name, i)).type(torch.long).squeeze().to(self.device)
                  # print("true",i,train_lbls)

                  idx_val = torch.load("./Experiment/sample_data/Node/{}/batch/{}/val_idx.pt".format(self.dataset_name, i)).type(torch.long).to(self.device)
                  val_lbls = torch.load("./Experiment/sample_data/Node/{}/batch/{}/val_labels.pt".format(self.dataset_name, i)).type(torch.long).squeeze().to(self.device)

                  idx_test = torch.load("./Experiment/sample_data/Node/{}/batch/{}/test_idx.pt".format(self.dataset_name, i)).type(torch.long).to(self.device)
                  test_lbls = torch.load("./Experiment/sample_data/Node/{}/batch/{}/test_labels.pt".format(self.dataset_name, i)).type(torch.long).squeeze().to(self.device)

                  #先从训练集中采样一个平衡分布，用于后续学习专家的权重
                  min_num = min(train_lbls.bincount())
                  min_num = min_num.cpu().numpy()

                  unique_labels = torch.unique(train_lbls)
                  # 用于保存所有的抽样下标
                  all_sampled_indices = []

                  for lbl in unique_labels:
                        # 找出每个标签的所有下标
                        indices = torch.where(train_lbls == lbl)[0]
                        # 随机打乱下标
                        indices = idx_train[indices]
                        shuffled_indices = indices[torch.randperm(indices.size(0))]
                        # 抽取前3个（如果标签的下标数不足min_num个，取所有下标）
                        sampled_indices = shuffled_indices[:min_num]
                        # 将每类标签的抽样下标添加到列表中
                        all_sampled_indices.append(sampled_indices)

                  fin_train_idx = torch.cat(all_sampled_indices)

                  shuffle_indices = torch.randperm(fin_train_idx.size(0))

                  # 根据这个随机排列打乱下标和对应的标签
                  fin_train_idx = fin_train_idx[shuffle_indices]

                  indices_2 = idx_train[train_lbls == 2]
                  indices_3 = idx_train[train_lbls == 3]

                  indices_0 = idx_train[train_lbls == 0]
                  indices_4 = idx_train[train_lbls == 4]

                  perm_2 = torch.randperm(len(indices_2))
                  perm_3 = torch.randperm(len(indices_3))

                  # 随机采样下标  53为躯干类的节点数量平均值，7为尾部类节点数量的平均值
                  sampled_indices_2for2 = indices_2[perm_2[:53]]
                  sampled_indices_3for2 = indices_3[perm_3[:53]]
                  idx_for2 = torch.cat((sampled_indices_2for2, sampled_indices_3for2), dim=0)

                  sampled_indices_2for3 = indices_2[perm_2[53:60]]
                  sampled_indices_3for3 = indices_3[perm_3[53:60]]
                  sampled_indices_0for3 = indices_0[torch.randperm(len(indices_0))[:7]]
                  sampled_indices_4for3 = indices_4[torch.randperm(len(indices_4))[:7]]

                  idx_for3 = torch.cat(  (sampled_indices_2for3, sampled_indices_3for3, sampled_indices_0for3, sampled_indices_4for3),  dim=0)

                  #将训练集进一步划分为头部类、躯干类、尾部类
                  train_head_idx = []
                  train_body_idx = []
                  train_tail_idx = []
                  for idx,l in zip(idx_train,train_lbls):
                        if l in head_label:
                              train_head_idx.append(idx.item())
                        elif l in body_label:
                              train_body_idx.append(idx.item())
                        elif l in tail_label:
                              train_tail_idx.append(idx.item())
                  train_head_idx = torch.tensor(train_head_idx)
                  train_body_idx = torch.tensor(train_body_idx)
                  train_tail_idx = torch.tensor(train_tail_idx)

                  # 将验证集进一步划分为头部类、躯干类、尾部类
                  val_head_idx = []
                  val_body_idx = []
                  val_tail_idx = []
                  for idx,l in zip(idx_val,val_lbls):
                        if l in head_label:
                              val_head_idx.append(idx.item())
                        elif l in body_label:
                              val_body_idx.append(idx.item())
                        elif l in tail_label:
                              val_tail_idx.append(idx.item())
                  val_head_idx = torch.tensor(val_head_idx)
                  val_body_idx = torch.tensor(val_body_idx)
                  val_tail_idx = torch.tensor(val_tail_idx)

                  # GPPT prompt initialtion
                  if self.prompt_type == 'GPPT':
                        node_embedding = self.gnn(self.data.x, self.data.edge_index)
                        self.prompt.weigth_init(node_embedding,self.data.edge_index, self.data.y, idx_train)


                  if self.prompt_type in ['Gprompt', 'All-in-one', 'GPF', 'GPF-plus']:
                        train_head_graphs = []
                        train_body_graphs = []
                        train_tail_graphs = []

                        val_head_graphs = []
                        val_body_graphs = []
                        val_tail_graphs = []

                        # 用于阶段四训练
                        fin_train_graphs = []

                        test_graphs = []

                        extra_graphfor_2 = []
                        extra_graphfor_3 = []
                        # self.graphs_list.to(self.device)
                        print('distinguishing the train dataset and test dataset...')

                        #记录用于最终训练的节点子图
                        for graph in self.graphs_list:
                              if graph.index in fin_train_idx:
                                    fin_train_graphs.append(graph)

                        for graph in self.graphs_list:
                              if graph.index in idx_for2:
                                    extra_graphfor_2.append(graph)
                              elif graph.index in idx_for3:
                                    extra_graphfor_3.append(graph)


                        for graph in self.graphs_list:
                              if graph.index in train_head_idx:
                                    train_head_graphs.append(graph)
                              elif graph.index in train_body_idx:
                                    train_body_graphs.append(graph)
                              elif graph.index in train_tail_idx:
                                    train_tail_graphs.append(graph)

                              elif graph.index in val_head_idx:
                                    val_head_graphs.append(graph)
                              elif graph.index in val_body_idx:
                                    val_body_graphs.append(graph)
                              elif graph.index in val_tail_idx:
                                    val_tail_graphs.append(graph)

                              elif graph.index in idx_test:
                                    test_graphs.append(graph)
                        print('Done!!!')

                        train_head_dataset = GraphDataset(train_head_graphs)
                        train_body_dataset = GraphDataset(train_body_graphs + extra_graphfor_2)
                        train_tail_dataset = GraphDataset(train_tail_graphs + extra_graphfor_3)

                        val_head_dataset = GraphDataset(val_head_graphs)
                        val_body_dataset = GraphDataset(val_body_graphs)
                        val_tail_dataset = GraphDataset(val_tail_graphs)

                        val_head_body_dataset = ConcatDataset([val_head_dataset, val_body_dataset])
                        val_all_dataset = ConcatDataset([val_head_dataset, val_body_dataset, val_tail_dataset])

                        fin_train_dataset = GraphDataset(fin_train_graphs)

                        test_dataset = GraphDataset(test_graphs)

                        bs = 128

                        # 创建数据加载器
                        train_head_loader = DataLoader(train_head_dataset, batch_size=bs, shuffle=True)
                        train_body_loader = DataLoader(train_body_dataset, batch_size=bs, shuffle=True)
                        train_tail_loader = DataLoader(train_tail_dataset, batch_size=bs, shuffle=True)

                        val_head_loader = DataLoader(val_head_dataset, batch_size=bs, shuffle=True)
                        val_head_body_loader = DataLoader(val_head_body_dataset, batch_size=bs, shuffle=True)
                        val_all_loader = DataLoader(val_all_dataset, batch_size=bs, shuffle=True)

                        # 用于训练混合专家
                        fin_train_loader = DataLoader(fin_train_dataset, batch_size=64, shuffle=False)

                        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
                        print("prepare induce graph data is finished!")

                  if self.prompt_type == 'MultiGprompt':
                        embeds, _ = self.Preprompt.embed(self.features, self.sp_adj, True, None, False)
                        pretrain_embs = embeds[0, idx_train]
                        test_embs = embeds[0, idx_test]

                  # --------------------------------------------------------------------------------------------------------------------------
                  # 开始一阶段

                  patience = 25
                  best = 0
                  cnt_wait = 0

                  # 构建保存路径
                  save_head_model_path = "./Experiment/downstream_model/{}/{}/{}/".format(self.dataset_name, 'model', batch_idx)
                  file_head_model_name = "{}.{}.{}.{}.pth".format('Edgepred_Gprompt', self.gnn_type, str(self.hid_dim) + 'hidden_dim', 'head')

                  save_head_prompt_path = "./Experiment/downstream_model/{}/{}/{}/".format(self.dataset_name, 'prompt',  batch_idx)
                  file_head_prompt_name = "{}.{}.{}.pth".format('Edgepred_Gprompt', 'prompt', 'head')

                  # 创建目录（如果不存在）
                  os.makedirs(save_head_model_path, exist_ok=True)
                  os.makedirs(save_head_prompt_path, exist_ok=True)
                  for epoch in range(1, self.epochs):
                        t0 = time.time()

                        if self.prompt_type == 'None':
                              loss = self.train(self.data, idx_train)
                        elif self.prompt_type == 'GPPT':
                              loss = self.GPPTtrain(self.data, idx_train)
                        elif self.prompt_type == 'All-in-one':
                              loss = self.AllInOneTrain(train_head_loader,self.answer_epoch,self.prompt_epoch)
                        elif self.prompt_type in ['GPF', 'GPF-plus']:
                              loss = self.GPFTrain(train_head_loader)
                        elif self.prompt_type =='Gprompt':
                              loss, center = self.GpromptTrain(train_head_loader)
                        elif self.prompt_type == 'MultiGprompt':
                              loss = self.MultiGpromptTrain(pretrain_embs, train_lbls, idx_train)

                        #开始验证
                        val_acc, f1, roc, prc = GpromptEva(val_head_loader, self.gnn, self.prompt, center, self.output_dim, self.device)

                        if best < val_acc:
                              best = val_acc
                              best_t = epoch
                              cnt_wait = 0

                              #保存head模型和prompt
                              torch.save(self.gnn.state_dict(),os.path.join(save_head_model_path, file_head_model_name))
                              print("+++model_head saved ! {}".format(save_head_model_path+file_head_model_name))

                              torch.save(self.prompt.state_dict(),os.path.join(save_head_prompt_path, file_head_prompt_name))
                              print("+++prompt_head saved ! {}".format(save_head_prompt_path+file_head_prompt_name))
                        else:
                              cnt_wait += 1
                              if cnt_wait == patience:
                                    print('-' * 100)
                                    print('Early stopping at '+str(epoch) +' eopch!')
                                    break

                        print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                        print("val_acc:{}".format(val_acc))

                  print("stage one finished！！！！！")

                  #--------------------------------------------------------------------------------------------------------------------------
                  #开始二阶段

                  patience = 25
                  best = 0
                  cnt_wait = 0

                  # 构建保存路径
                  save_body_model_path = "./Experiment/downstream_model/{}/{}/{}/".format(self.dataset_name, 'model',batch_idx)
                  file_body_model_name = "{}.{}.{}.{}.pth".format('Edgepred_Gprompt', self.gnn_type,str(self.hid_dim) + 'hidden_dim', 'body')

                  save_body_prompt_path = "./Experiment/downstream_model/{}/{}/{}/".format(self.dataset_name, 'prompt',batch_idx)
                  file_body_prompt_name = "{}.{}.{}.pth".format('Edgepred_Gprompt', 'prompt', 'body')

                  self.gnn.load_state_dict(torch.load(os.path.join(save_head_model_path, file_head_model_name),map_location='cpu'))
                  self.gnn.to(self.device)
                  print("Successfully loaded head_model weights!")


                  self.prompt.load_state_dict(torch.load(os.path.join(save_head_prompt_path, file_head_prompt_name),map_location='cpu'))
                  self.prompt.to(self.device)
                  print("Successfully loaded head_prompt weights!")
                  self.initialize_optimizer()

                  for epoch in range(1, self.epochs):
                        t0 = time.time()
                        loss, center = self.GpromptTrain(train_body_loader)

                        # 开始验证
                        val_acc_second, f1, roc, prc = GpromptEva(val_head_body_loader, self.gnn, self.prompt, center,
                                                           self.output_dim, self.device)

                        if best < val_acc_second:
                              best = val_acc_second
                              best_t = epoch
                              cnt_wait = 0

                              # 保存body模型和prompt
                              torch.save(self.gnn.state_dict(),os.path.join(save_body_model_path, file_body_model_name))
                              print("+++model_body saved ! {}".format(save_body_model_path + file_body_model_name))

                              torch.save(self.prompt.state_dict(),os.path.join(save_body_prompt_path, file_body_prompt_name))
                              print("+++prompt_body saved ! {}".format(save_body_prompt_path + file_body_prompt_name))
                        else:
                              cnt_wait += 1
                              if cnt_wait == patience:
                                    print('-' * 100)
                                    print('Early stopping at ' + str(epoch) + ' eopch!')
                                    break

                        print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                        print("val_acc_second:{}".format(val_acc_second))

                  print("stage two finished！！！！！")

                  # --------------------------------------------------------------------------------------------------------------------------
                  # 开始三阶段

                  patience = 25
                  best = 0
                  cnt_wait = 0

                  # 构建保存路径
                  save_tail_model_path = "./Experiment/downstream_model/{}/{}/{}/".format(self.dataset_name, 'model',batch_idx)
                  file_tail_model_name = "{}.{}.{}.{}.pth".format('Edgepred_Gprompt', self.gnn_type,str(self.hid_dim) + 'hidden_dim', 'tail')

                  save_tail_prompt_path = "./Experiment/downstream_model/{}/{}/{}/".format(self.dataset_name, 'prompt', batch_idx)
                  file_tail_prompt_name = "{}.{}.{}.pth".format('Edgepred_Gprompt', 'prompt', 'tail')

                  self.gnn.load_state_dict(torch.load(os.path.join(save_body_model_path, file_body_model_name), map_location='cpu'))
                  self.gnn.to(self.device)
                  print("Successfully loaded body_model weights!")

                  self.prompt.load_state_dict(torch.load(os.path.join(save_body_prompt_path, file_body_prompt_name), map_location='cpu'))
                  self.prompt.to(self.device)
                  print("Successfully loaded body_prompt weights!")
                  self.initialize_optimizer()

                  for epoch in range(1, self.epochs):
                        t0 = time.time()
                        loss, center = self.GpromptTrain(train_tail_loader)

                        # 开始验证
                        val_acc_third, f1, roc, prc = GpromptEva(val_all_loader, self.gnn, self.prompt, center,
                                                                  self.output_dim, self.device)

                        if best < val_acc_third:
                              best = val_acc_third
                              best_t = epoch
                              cnt_wait = 0

                              # 保存tail模型和prompt
                              torch.save(self.gnn.state_dict(),os.path.join(save_tail_model_path, file_tail_model_name))
                              print("+++model_tail saved ! {}".format(save_tail_model_path + file_tail_model_name))

                              torch.save(self.prompt.state_dict(),os.path.join(save_tail_prompt_path, file_tail_prompt_name))
                              print("+++prompt_tail saved ! {}".format(save_tail_prompt_path + file_tail_prompt_name))
                        else:
                              cnt_wait += 1
                              if cnt_wait == patience:
                                    print('-' * 100)
                                    print('Early stopping at ' + str(epoch) + ' eopch!')
                                    break

                        print("Epoch {:03d} |  Time(s) {:.4f} | Loss {:.4f}  ".format(epoch, time.time() - t0, loss))
                        print("val_acc_third:{}".format(val_acc_third))
                  #存放原型向量
                  centers[f'center_{batch_idx}'] = center
                  with open('centers.pkl', 'wb') as f:
                        pickle.dump(centers, f)

                  print("stage three finished！！！！！")

                  # --------------------------------------------------------------------------------------------------------------------------
                  # 开始四阶段，训练专家
                  with open('centers.pkl', 'rb') as f:
                        centers = pickle.load(f)
                  all_center = centers[f'center_{batch_idx}']

                  #加载参数
                  self.gnn_head.load_state_dict(torch.load(os.path.join(save_head_model_path, file_head_model_name), map_location='cpu'))
                  self.gnn_head.to(self.device)
                  self.head_prompt.load_state_dict(torch.load(os.path.join(save_head_prompt_path, file_head_prompt_name), map_location='cpu'))
                  self.head_prompt.to(self.device)

                  self.gnn_body.load_state_dict(torch.load(os.path.join(save_body_model_path, file_body_model_name), map_location='cpu'))
                  self.gnn_body.to(self.device)
                  self.body_prompt.load_state_dict(torch.load(os.path.join(save_body_prompt_path, file_body_prompt_name), map_location='cpu'))
                  self.body_prompt.to(self.device)

                  self.gnn_tail.load_state_dict(torch.load(os.path.join(save_tail_model_path, file_tail_model_name), map_location='cpu'))
                  self.gnn_tail.to(self.device)
                  self.tail_prompt.load_state_dict(torch.load(os.path.join(save_tail_prompt_path, file_tail_prompt_name), map_location='cpu'))
                  self.tail_prompt.to(self.device)

                  w_head = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
                  w_body = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
                  w_tail = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

                  optimizer = optim.Adam([w_head, w_body, w_tail], lr=0.03)
                  criterion = nn.CrossEntropyLoss()

                  print("stage four start！！！！！")

                  for epoch in range(300):
                        for batch_id, batch in enumerate(fin_train_loader):
                              batch = batch.to(self.device)
                              # 获取每个专家的输出
                              out_head = self.gnn_head(batch.x, batch.edge_index, batch.batch, self.head_prompt,'Gprompt')
                              out_body = self.gnn_body(batch.x, batch.edge_index, batch.batch, self.body_prompt,'Gprompt')
                              out_tail = self.gnn_tail(batch.x, batch.edge_index, batch.batch, self.tail_prompt,'Gprompt')

                              # 计算相似度矩阵
                              head_similarity_matrix = F.cosine_similarity(out_head.unsqueeze(1), all_center.unsqueeze(0), dim=-1)
                              body_similarity_matrix = F.cosine_similarity(out_body.unsqueeze(1),all_center.unsqueeze(0), dim=-1)
                              tail_similarity_matrix = F.cosine_similarity(out_tail.unsqueeze(1),all_center.unsqueeze(0), dim=-1)

                              optimizer.zero_grad()

                              # 对三个专家的预测结果进行加权求和
                              weighted_pred = w_head * head_similarity_matrix + w_body * body_similarity_matrix + w_tail * tail_similarity_matrix
                              pred = weighted_pred.argmax(dim=1)

                              # 计算损失
                              loss = criterion(weighted_pred, batch.y)

                              # 反向传播并优化权重
                              loss.backward()
                              optimizer.step()

                  print("stage four finished！！！！！")
                  # --------------------------------------------------------------------------------------------------------------------------
                  # 开始测试！！！！！！！！！
                  test_acc, f1, roc, prc = GpromptEva_new(test_loader, self.gnn_head, self.head_prompt,self.gnn_body, self.body_prompt,self.gnn_tail, self.tail_prompt,
                                                      w_head,w_body,w_tail,all_center, self.output_dim, self.device)
                  print(f"Final True Accuracy: {test_acc:.4f} | Macro F1 Score: {f1:.4f} | AUROC: {roc:.4f} | AUPRC: {prc:.4f}")
                  test_accs.append(test_acc)
                  f1s.append(f1)
                  rocs.append(roc)
                  prcs.append(prc)

            mean_test_acc = np.mean(test_accs)
            std_test_acc = np.std(test_accs)
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            mean_roc = np.mean(rocs)
            std_roc = np.std(rocs)
            mean_prc = np.mean(prcs)
            std_prc = np.std(prcs)
            print(" Final best | test Accuracy {:.4f}±{:.4f}(std)".format(mean_test_acc, std_test_acc))
            print(" Final best | test F1 {:.4f}±{:.4f}(std)".format(mean_f1, std_f1))
            print(" Final best | AUROC {:.4f}±{:.4f}(std)".format(mean_roc, std_roc))
            print(" Final best | AUPRC {:.4f}±{:.4f}(std)".format(mean_prc, std_prc))
            print('test_accs:',test_accs)
            print('f1s:',f1s)
            print('rocs:',rocs)
            print('prcs:',prcs)
            print(self.pre_train_type, self.gnn_type, self.prompt_type, " Graph Task completed")
            return mean_test_acc, std_test_acc, mean_f1, std_f1, mean_roc, std_roc, mean_prc, std_prc

