import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
from model.attention import MultiheadAttention
from model.gvp_layers import GVP, LayerNorm



class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, args, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0.,):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.use_gvp = args.gvp

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25

        if not self.use_gvp:
            self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
            self.norm_edges = nn.LayerNorm(edge_features)
        else:
            self.GVP_v = nn.Sequential(
                GVP((6, 2), (80, 16), activations=(None, None)),
                LayerNorm((80, 16)),
                GVP((80, 16), (80, 16), activations=(None, None)),
                LayerNorm((80, 16)),
            )
            self.GVP_e = nn.Sequential(
                GVP((edge_in, 1), (125, 1), activations=(None, None)),
                LayerNorm((125, 1)),
                GVP((125, 1), (125, 1), activations=(None, None)),
                LayerNorm((125, 1)),
            )

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D, D_min = 2., D_max = 22.):
        device = D.device
        D_count = self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        d_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), d_chains)
        E = torch.cat((E_positional, RBF_all), -1)
    
        h_V = torch.zeros((E.shape[0], E.shape[1], self.edge_features), device = E.device)

        E_s = E
        Ca_to = Ca.view(Ca.shape[0], 1, Ca.shape[1], 3).expand(Ca.shape[0], Ca.shape[1], Ca.shape[1], 3)
        idx = E_idx.view(*E_idx.shape, 1).expand(*E_idx.shape, 3)
        Ca_to = torch.gather(Ca_to, 2, idx)
        Ca_fr = Ca.view(Ca.shape[0], Ca.shape[1], 1, 3).expand(Ca.shape[0], Ca.shape[1], Ca_to.shape[2], 3)
        E_v = F.normalize(Ca_to - Ca_fr, p = 2, dim = -1).unsqueeze(-2)

        N_s = self._dihedrals(X)
        N_v = self._orientations(Ca)

        N_s, N_v = self.GVP_v((N_s, N_v))
        E_s, E_v = self.GVP_e((E_s, E_v))
        h_V = torch.cat([N_s, N_v.reshape(*N_v.shape[:2], -1)], dim = -1)
        E = torch.cat([E_s, E_v.reshape(*E_v.shape[:3], -1)], dim = -1)

        return h_V, E, E_idx 

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def _orientations(self, X):
        forward = F.normalize(X[:, 1:] - X[:, :-1], p = 2, dim = -1)
        backward = F.normalize(X[:, :-1] - X[:, 1:], p = 2, dim = -1)
        forward = F.pad(forward, [0, 0, 0, 1, 0, 0])
        backward = F.pad(backward, [0, 0, 1, 0, 0, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, args, d_model, d_node, n_head, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiheadAttention(args, d_model, d_model, n_head, d_node, d_node, d_node, dropout = dropout)
        
        self.non_edge_E = nn.Parameter(torch.randn(d_model))
        self.atten_bias = nn.Linear(d_model, 1)

        self.W11 = nn.Linear(d_model * 3, d_model, bias=True)
        self.W12 = nn.Linear(d_model, d_model, bias=True)
        self.W13 = nn.Linear(d_model, d_model, bias=True)
        self.act = torch.nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h_V, E, E_idx, slf_attn_mask = None, edge_mask = None):
        B, L = h_V.shape[0], h_V.shape[1]

        bias = torch.ones(B, L, L, device = h_V.device) * self.atten_bias(self.non_edge_E)
        bias_edge = self.atten_bias(E)
        bias = torch.scatter(bias, -1, E_idx, bias_edge.view(B, L, -1))
        h_V = self.slf_attn(h_V, h_V, h_V, E, E_idx, bias, mask = slf_attn_mask) * slf_attn_mask.unsqueeze(-1)

        h_EV = cat_neighbors_nodes(h_V, E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm(E + self.dropout(h_message))

        return h_V, h_E


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, args, n_layers, n_head, d_model, dropout = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(p = dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(args, d_model, d_model * 2 if i == 0 else d_model, n_head, dropout = dropout)
            for i in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model * 2, eps = 1e-6)

    def forward(self, h_V, E, E_idx, src_mask, edge_mask = None):
        h_V = self.dropout(h_V)
        h_V = self.layer_norm(h_V)

        for enc_layer in self.layer_stack:
            h_V, E = enc_layer(h_V, E, E_idx, slf_attn_mask = src_mask, edge_mask = edge_mask)

        return h_V, E


class Model(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, args, k_neighbors, n_head = 8, vocab = 21):

        super().__init__()


        self.hidden_dim = args.hidden_dim
        self.n_layers = args.trans_layers
        self.th = args.th
        self.seq_noise = args.seq_noise
        self.in_dim = args.in_dim
        
        self.k_neighbors = k_neighbors
        dropout = args.dropout

        self.features = ProteinFeatures(args, self.hidden_dim, self.hidden_dim, top_k = k_neighbors, augment_eps = args.backbone_noise)

        self.s_embed = nn.Embedding(vocab + 1, self.hidden_dim)
        self.encoder = Encoder(args, self.n_layers, n_head, self.hidden_dim, dropout = dropout)

        self.mlp_nodes = nn.Linear(self.hidden_dim, vocab)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 


    def forward(self, X, S, mask, residue_idx, chain_encoding_all, mask_visible = None, feat = False):
        S = S.clone()
        h_V, h_E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)

        if self.training and self.seq_noise > 0.:
            noise_mask = torch.rand_like(mask) < self.seq_noise
            S[noise_mask] = torch.randint(0, 21, S.shape, device = S.device, dtype = S.dtype)[noise_mask]
        if self.training:
            mask_en = torch.rand_like(mask) < self.th
        elif mask_visible is None:
            mask_en = torch.zeros_like(mask, dtype = torch.bool)
        else:
            mask_en = mask_visible.bool()
        S[~mask_en] = 21

        h_S = self.s_embed(S)
        edge_mask = None
        h_V, h_E = self.encoder(torch.cat([h_V, h_S], dim = -1), h_E, E_idx, mask, edge_mask = edge_mask)

        logits = self.mlp_nodes(h_V)
        log_probs = F.log_softmax(logits, dim=-1) 

        if feat:
            return log_probs, h_V
        else:
            return log_probs