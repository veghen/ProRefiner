import torch


model = None


def main(batch, device, temperature, mask_visible = None, S_env = None):

    import json, os
    import numpy as np
    import torch
    import os.path
    
    from ProteinMPNN.protein_mpnn_utils import tied_featurize
    from ProteinMPNN.protein_mpnn_utils import ProteinMPNN

    
    hidden_dim = 128
    num_layers = 3 

    checkpoint = torch.load("ProteinMPNN/v_48_020.pt", map_location=device) 
    args = checkpoint["args"]
  

    if args.path_to_model_weights:
        model_folder_path = args.path_to_model_weights
        if model_folder_path[-1] != '/':
            model_folder_path = model_folder_path + '/'
    else: 
        file_path = os.path.realpath(__file__)
        k = file_path.rfind("/")
        if args.ca_only:
            model_folder_path = file_path[:k] + '/ca_model_weights/'
        else:
            model_folder_path = file_path[:k] + '/vanilla_model_weights/'


    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        
    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        fixed_positions_dict = None
    
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        pssm_dict = None
    
    
    if os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        omit_AA_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        bias_AA_dict = None
    
    
    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        tied_positions_dict = None

    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
    else:
        bias_by_res_dict = None
   
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]

    global model
    if model is None:
        model = ProteinMPNN(ca_only=args.ca_only, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'])
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    X, _, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=args.ca_only)
    pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float() #1.0 for true, 0.0 for false

    if mask_visible is None:
        S_env = torch.zeros_like(mask, dtype = torch.long)
        mask_visible = torch.zeros_like(mask)
    mask_visible = mask_visible.float()

    if args.score_only or args.conditional_probs_only or args.unconditional_probs_only:
        assert 0
    else:
        randn_2 = torch.randn(chain_M.shape, device=X.device)
        sample_dict = model.sample(X, randn_2, S_env, chain_M * (1 - mask_visible), chain_encoding_all, residue_idx, mask=mask, temperature=temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=args.pssm_multi, pssm_log_odds_flag=bool(args.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(args.pssm_bias_flag), bias_by_res=bias_by_res_all)

        S_sample = sample_dict["S"]
        log_probs = model(X, S_sample, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_2, use_input_decoding_order=True, decoding_order=sample_dict["decoding_order"])

    return S_sample, log_probs

   
def run(batch, device, temperature, mask_visible = None, S_env = None):
    torch.set_grad_enabled(False)
    return main(batch, device, temperature, mask_visible = mask_visible, S_env = S_env)