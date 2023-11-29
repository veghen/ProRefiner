import os
import requests
import numpy as np
import torch
from Bio.PDB import PDBParser


AA_MAP = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
alphabet = 'ACDEFGHIKLMNPQRSTVWYX'


def parse_chain(c, b_factor = False, v = False):
    seq = ''
    coords = {"N": [], "CA": [], "C": [], "O": []}
    bf = []
    for r in c:
        r_id = r.get_id()
        if r_id[0] != ' ':
            continue
        if r.get_resname() not in AA_MAP:
            return False, '', {}
            
        seq += AA_MAP[r.get_resname()]

        for k in coords.keys():
            try:
                coord = r[k].get_coord()
            except:
                coord = np.array([np.nan, np.nan, np.nan])
                if v: print(r.get_full_id(), k)
            coords[k].append(coord)
        
        if b_factor:
            bf.append(r["CA"].get_bfactor())

    if b_factor:
        return True, seq, coords, bf
    return True, seq, coords


def get_pdb(pdb_code_or_path, chain = "A"):
    if os.path.exists(pdb_code_or_path):
        pdb_code = pdb_code_or_path
        pdb_path = pdb_code_or_path
    else:
        pdb_code = pdb_code_or_path
        pdb_path = pdb_code + ".pdb"
        r = requests.get("https://files.rcsb.org/view/{}.pdb".format(pdb_code))
        if r.status_code == 200:
            open(pdb_path, 'wb').write(r.content)
        else:
            print("Please provide a valid PDB code or PDB file path.")
            exit(0)

    p = PDBParser(PERMISSIVE=1)
    s = p.get_structure(pdb_code, pdb_path)

    # get the chain from the first model
    try:
        c = s[0][chain]
    except:
        chain_list = [c.get_id() for c in s[0].get_chains()]
        print("Chain {} not found in {} (chains: {})".format(chain, pdb_code, chain_list))
        exit(0)
        
    flag, seq, coords = parse_chain(c)
    assert flag, pdb_code
    
    d = {}
    d["name"] = pdb_code
    d["seq"] = seq
    d["coords"] = coords
    assert len(seq) == len(coords["CA"])

    print("Read {} chain {} with length {}".format(pdb_code, chain, len(seq)))

    return d


def get_entropy(log_probs):
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -1 * p_log_p.mean(dim = -1)
    return entropy


def fuse_log_probs(log_prob_list, temp = 1.):
    entropy_list = [get_entropy(log_probs) for log_probs in log_prob_list]
    entropy = torch.stack(entropy_list, dim = 0)
    entropy = torch.nn.functional.softmax(-1 * entropy / temp, dim = 0)

    # fuse by entropy
    if type(log_prob_list) is list:
        log_prob_list = torch.stack(log_prob_list, dim = 0)
    log_prob = (entropy.unsqueeze(-1) * log_prob_list).sum(dim = 0)

    return log_prob


def compute_rec(S_pred, S_gt, mask):
    return (((S_pred == S_gt) * mask).sum(dim = -1) / mask.sum(dim = -1)).item()


def compute_nssr(S_pred, S_gt, mask):
    B_mat = torch.load("B_mat.pth").to(mask.device)
    scores = B_mat[S_pred, S_gt] > 0
    nssr = ((scores * mask).sum(dim = -1) / mask.sum(dim = -1)).item()
    return nssr


def get_features(batch, device, shuffle_fraction=0., crop_len = 9999999):
    """ Pack and pad batch into torch tensors """
    B = len(batch)
    lengths = np.array([min(len(b['seq']), crop_len) for b in batch], dtype=np.int32)
    L_max = max([min(len(b['seq']), crop_len) for b in batch])
    X = np.zeros([B, L_max, 4, 3])
    S = np.zeros([B, L_max], dtype=np.int32)
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)

    def shuffle_subset(n, p):
        n_shuffle = np.random.binomial(n, p)
        ix = np.arange(n)
        ix_subset = np.random.choice(ix, size=n_shuffle, replace=False)
        ix_subset_shuffled = np.copy(ix_subset)
        np.random.shuffle(ix_subset_shuffled)
        ix[ix_subset] = ix_subset_shuffled
        return ix

    # Build the batch
    for i, b in enumerate(batch):
        x = np.stack([b['coords'][c] for c in ['N', 'CA', 'C', 'O']], 1)
        
        l = len(b['seq'])
        crop_s = None
        if l > crop_len:
            crop_s = np.random.randint(0, high = l - crop_len + 1)
            x = x[crop_s : crop_s + crop_len]
            l = crop_len
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad
        residue_idx[i, 0 : l] = np.arange(0, l)
        chain_encoding_all[i, 0 : l] = np.ones(l)

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in b['seq']], dtype=np.int32)
        if crop_s is not None:
            indices = indices[crop_s : crop_s + crop_len]
        if shuffle_fraction > 0.:
            idx_shuffle = shuffle_subset(l, shuffle_fraction)
            S[i, :l] = indices[idx_shuffle]
        else:
            S[i, :l] = indices

    # Mask
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    S = torch.from_numpy(S).to(dtype = torch.long,device = device)
    X = torch.from_numpy(X).to(dtype = torch.float32, device = device)
    mask = torch.from_numpy(mask).to(dtype = torch.float32, device = device)
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)    
    lengths = torch.from_numpy(lengths).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, residue_idx, chain_encoding_all


def tostr(S):
    return ''.join([alphabet[i] for i in S])


# from ProteinMPNN
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.reshape((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn
