import warnings
warnings.filterwarnings("ignore")
import sys

import torch

from ProteinMPNN.proteinmpnn import run as run_proteinmpnn
from model.model import Model
from utils import *


def run_one_batch_partial(batch, device, design_shell):
    '''
    design shell: list of residues to be designed, index starting from 1
    '''
    X, S_gt, mask, _, residue_idx, chain_encoding_all = get_features(batch, device)

    S_env = torch.zeros_like(S_gt) - 1
    mask_design = torch.zeros_like(mask)
    design_shell = torch.tensor(design_shell, device = device) - 1
    mask_design[0, design_shell] = 1.
    mask_design = mask_design * mask
    S_env[((1 - mask_design) * mask).bool()] = S_gt[((1 - mask_design) * mask).bool()]

    S_sample, _ = run_proteinmpnn(batch, device, 1e-3, mask_visible = (1 - mask_design) * mask, S_env = torch.clamp(S_env, min = 0))
    log_probs = model(X, torch.clamp(S_env, min = 0), mask, residue_idx, chain_encoding_all, mask_visible = (1 - mask_design) * mask)
    

    return S_gt, S_sample, torch.argmax(log_probs, dim = -1), mask_design.bool()


def run_one_batch_entire(batch, device):
    X, S_gt, mask, _, residue_idx, chain_encoding_all = get_features(batch, device)
    mask_design = mask
    
    S_sample, log_probs_base = run_proteinmpnn(batch, device, 1e-3, mask_visible = torch.zeros_like(mask), S_env = torch.zeros_like(S_gt))

    th = 0.1
    entropy = get_entropy(log_probs_base)
    mask_visible = ((entropy < torch.quantile(entropy[mask.bool()], th)) * mask).bool()

    S = torch.argmax(log_probs_base, dim = -1)
    S_env = torch.zeros_like(S_gt) - 1
    S_env[mask_visible] = S[mask_visible]
    
    log_probs = model(X, torch.clamp(S_env, min = 0), mask, residue_idx, chain_encoding_all, mask_visible = (S_env > -1) * mask)
    log_probs = fuse_log_probs([log_probs_base, log_probs])

    return S_gt, S_sample, torch.argmax(log_probs, dim = -1), mask_design.bool()


def run_one_batch(batch, device, design_shell):
    if len(design_shell) == 0:
        return run_one_batch_entire(batch, device)
    else:
        return run_one_batch_partial(batch, device, design_shell)



if __name__ == "__main__":
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    
    # load model
    checkpoint = torch.load("model/checkpoint.pth", map_location = device)
    model = Model(checkpoint["args"], 30, n_head = checkpoint["args"].encoder_attention_heads).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # parse pdb and design shell
    pdb_code = sys.argv[1]
    chain_code = (sys.argv[2]).upper()
    data = [get_pdb(pdb_code, chain_code), ]
    if len(sys.argv) > 3:
        design_shell = sys.argv[3]
        design_shell = [int(i) for i in design_shell.strip().split(',')]
    else:
        design_shell = []

    our_model_name = "ProRefiner + ProteinMPNN" if len(design_shell) == 0 else "ProRefiner"

    # run design sequence
    S_gt, S_base, S, mask_design = run_one_batch(data, device, design_shell)

    seq_recovery_rate_bl = compute_rec(S_base, S_gt, mask_design)
    nssr_bl = compute_nssr(S_base, S_gt, mask_design)

    seq_recovery_rate = compute_rec(S, S_gt, mask_design)
    nssr = compute_nssr(S, S_gt, mask_design)

    print("\nDesign {} residues from {} chain {} (ignore residues without coordinates)\n".format(mask_design.sum().item(), pdb_code, chain_code))
    print("native sequence:")
    print(tostr(S_gt[mask_design]))
    print("\nsequence by ProteinMPNN: (recovery: {:.3f}\tnssr: {:.3f})".format(seq_recovery_rate_bl * 100., nssr_bl * 100.))
    print(tostr(S_base[mask_design]))
    print("\nsequence by {}: (recovery: {:.3f}\tnssr: {:.3f})".format(our_model_name, seq_recovery_rate * 100., nssr * 100.))
    print(tostr(S[mask_design]))