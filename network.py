import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from pool import GraphMultisetTransformer
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_max_pool as gmp

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True), nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.mlp(x)


class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[512, 512, 512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN, self).__init__()

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims

        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512, 256, 512, None, 10000, 0.25, ['GMPool_G', 'GMPool_G'], num_heads=8, layer_norm=True)
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2)
    
    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, data, pertubed=False):
        #x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            elif idx == 2:
                with torch.enable_grad():
                    self.final_conv_acts = gcn_layer(x, data.edge_index.long())
                self.final_conv_acts.register_hook(self.activations_hook)
                x = x + F.relu(self.final_conv_acts)
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long()))
            
            if pertubed:
                random_noise = torch.rand_like(x).to(x.device)
                x = x + torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x
#         g_level_feat = self.drop2(x)
#         # Compute fully-connected part
#         if self.fc_dim > 0:
#             g_level_feat = F.relu(self.fc(g_level_feat))

#         g_level_feat = self.fc_out(g_level_feat)

        return n_level_feat, g_level_feat


class CL_protNET(torch.nn.Module):
    def __init__(self, out_dim, af_embed=True, pooling='MTP', pertub=False):
        super(CL_protNET,self).__init__()
        self.af_embed = af_embed
        self.pertub = pertub
        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96, 512) 
        self.pooling = pooling
        #self.proj_spot = nn.Linear(19, 512)
        if af_embed:
            self.proj_esm = nn.Linear(1280, 512)
            self.gcn = GraphCNN(pooling=pooling)
        else:
            self.gcn = GraphCNN(pooling=pooling)
        #self.esm_g_proj = nn.Linear(1280, 512)
        self.readout = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(1024, out_dim),
                        nn.Sigmoid()
        )
        
        self.softmax = nn.Softmax(dim=-1)
        #self.logsoftmax = nn.LogSoftmax(dim=-1)       
    def forward(self, data):

        #x = data.x.float()
        #esm_graph_emb = gmp(x, data.batch) 
        #esm_graph_emb = self.esm_g_proj(esm_graph_emb)
        x_aa = self.one_hot_embed(data.native_x.long())
        x_aa = self.proj_aa(x_aa)
        #x_spot = self.proj_spot(data.spot_x)
        
        if self.af_embed:
            x = data.x.float()
            x_esm = self.proj_esm(x)
            x = F.relu(x_aa + x_esm)
            
        else:
            x = F.relu(x_aa)

        #x_cnn, mask = to_dense_batch(x, data.batch)
        gcn_n_feat1, gcn_g_feat1 = self.gcn(x, data)
        if self.pertub:
            gcn_n_feat2, gcn_g_feat2 = self.gcn(x, data, pertubed=True) 

            #gcn_g_feat1 = F.relu(gcn_g_feat1 + esm_graph_emb)
            y_pred = self.readout(gcn_g_feat1)
            #y_pred = y_pred.reshape([-1,self.out_dim,2])
            #y_pred = self.softmax(y_pred)
            return y_pred, gcn_g_feat1, gcn_g_feat2
        else:
            #gcn_g_feat1 = F.relu(gcn_g_feat1)
            y_pred = self.readout(gcn_g_feat1)
            #y_pred = y_pred.reshape([-1,self.out_dim,2]) 
            return y_pred

class CNN1D(nn.Module):
    def __init__(self, input_dim=384+64, num_filters=8*[256,512], filter_sizes=list(range(8,129,8)), out_dim=512):
        super(CNN1D, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        # Define 1D convolutional layers
        cnn_layers = []
        for i in range(len(num_filters)//2):
            if i == 0:
                cnn_layers.append(nn.Sequential(nn.Conv1d(input_dim, num_filters[2*i], kernel_size=filter_sizes[2*i]//2-1, padding=(filter_sizes[2*i]//2-2)//2),
                              nn.ReLU(),     
                              nn.InstanceNorm1d(num_filters[2*i]),
                              nn.Dropout(0.3), 
                              nn.Conv1d(num_filters[2*i], num_filters[2*i+1], kernel_size=filter_sizes[2*i+1]//2-1, padding=(filter_sizes[2*i+1]//2-2)//2),
                              nn.ReLU(),     
                              nn.InstanceNorm1d(num_filters[2*i+1]),
                              nn.Dropout(0.3), 
                                  ))
            else:
                cnn_layers.append(nn.Sequential(nn.Conv1d(num_filters[2*i-1], num_filters[2*i], kernel_size=filter_sizes[2*i]//2-1, padding=(filter_sizes[2*i]//2-2)//2),
                              nn.ReLU(),     
                              nn.InstanceNorm1d(num_filters[2*i]),
                              nn.Dropout(0.3),
                              nn.Conv1d(num_filters[2*i], num_filters[2*i+1], kernel_size=filter_sizes[2*i+1]//2-1, padding=(filter_sizes[2*i+1]//2-2)//2),
                              nn.ReLU(),     
                              nn.InstanceNorm1d(num_filters[2*i+1]),
                              nn.Dropout(0.3),
                                  ))                 
        self.cnn = nn.ModuleList(cnn_layers)

        # Define global max pooling
#         pool_layers = [nn.AdaptiveMaxPool1d(1) for _ in num_filters]
#         self.globalpool = nn.ModuleList(pool_layers)

        # Define fully-connected layers
        self.fc_out = nn.Linear(num_filters[-1], out_dim)

    def forward(self, x, mask):
#         x = self.dropout1(x.float())
        x = x.float().permute(0, 2, 1)
        # Compute 1D convolutional part and apply global max pooling
        for idx, cnn_layer in enumerate(self.cnn):
            if idx == 0:
                out = self.cnn[idx](x)
            else:
                out = out + cnn_layer(out)
                
        # Concatenate all channels and flatten vector
        n_level_feat = out.permute(0, 2, 1)
        #g_level_feat = torch.sum(n_level_feat * mask.unsqueeze(-1), -2) / mask.sum(-1).unsqueeze(-1)
        g_level_feat = out.max(dim=-1)[0]
        n_level_feat = n_level_feat.reshape(-1, 512)[mask.flatten()]
        # Compute fully-connected part
        
        g_level_feat = self.fc_out(g_level_feat)  

        return n_level_feat, g_level_feat


class CNN1D_DeepGoPlus(nn.Module):
    def __init__(self, num_classes, input_dim=26, num_filters=16*[512], filter_sizes=list(range(8,129,8))):
        super(CNN1D_DeepGoPlus, self).__init__()
        
        # Define 1D convolutional layers
        cnn_layers = [nn.Conv1d(input_dim, num_filters[i], kernel_size=filter_sizes[i], padding=int(filter_sizes[i]/2)-1)
                      for i in range(len(num_filters))]
        self.cnn = nn.ModuleList(cnn_layers)

        # Define global max pooling
        pool_layers = [nn.AdaptiveMaxPool1d(1) for _ in num_filters]
        self.globalpool = nn.ModuleList(pool_layers)

        # Define fully-connected layers
        self.fc_out = nn.Sequential(
                      nn.Linear(sum(num_filters), num_classes),
                      nn.Sigmoid()
                        )

    def forward(self, data):
        with torch.no_grad():
            x = F.one_hot(data.native_x.long(), 26).float()
        x, mask = to_dense_batch(x, data.batch)
        
        x = x.permute(0, 2, 1)
        # Compute 1D convolutional part and apply global max pooling
        all_x = []
        for cnn_layer, pool_layer in zip(self.cnn, self.globalpool):
            all_x.append(pool_layer(cnn_layer(x)))

        # Concatenate all channels and flatten vector
        x = torch.cat(all_x, dim=1)
        x = torch.flatten(x, 1)
        #embedding = x
        
        # Compute fully-connected part
        output = self.fc_out(x)   # sigmoid in loss function

        return output