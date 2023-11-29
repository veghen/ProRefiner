# ProRefiner: An Entropy-based Refining Strategy for Inverse Protein Folding with Global Graph Attention

This is a demo code for paper ProRefiner: An Entropy-based Refining Strategy for Inverse Protein Folding with Global Graph Attention. You can also run the demo online through [Colab](https://colab.research.google.com/drive/1a6VW-BB0twEwL65sE_dUAM42wdSm6RZp?usp=sharing) or [Code Ocean](https://codeocean.com/capsule/9492154/tree) for easier environment setup. 

## File structure
We provide the ProRefiner implementation in folder `model`. We put the code provided by [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) in folder `ProteinMPNN`. `run.py` contains the sequence design pipeline.


## Environemnt setup
The program is written in Python. Please first install [Python](https://www.python.org/downloads/) on the machine. Then run the following script in the terminal to setup the environment. It will automatically install the latest version of the packages.

    pip install torch torchvision torchaudio
    pip install biopython
    pip install fairseq

We recommend running on Linux systems. The code has been tested on the latest version of the above dependencies. The setup should be completed within few minutes.

## Run protein sequence design
This demo demonstrates sequence design with base model ProteinMPNN. Full sequence design and partial sequence design are supported. Designing one protein is fast with few seconds on CPUs.

### Full sequence design
Run the following script to start design.

    python run.py PDB_CODE_or_PDB_PATH CHAIN

For example:
    
    python run.py 8flh A                # provide a PDB code
    python run.py input/8flh.pdb A      # provide a PDB file path 

If a PDB code is provided, the PDB file will be automatically downloaded. The program will run sequence design on the specified chain (only single chain design is supported). Here is an example output of script `python run.py 8flh A`:

    Read 8flh chain A with length 266
    Recovery and nssr are computed based on designable residues with coordinates (265 residues).

    native sequence:
    YGSWEIDPKDLTFLKELGTGQFGVVKYGKWRGQYDVAIKMIKEGSMSEDEFIEEAKVMMNLSHEKLVQLYGVCTKQRPIFIITEYMANGCLLNYLREMRHRFQTQQLLEMCKDVCEAMEYLESKQFLHRDLAARNCLVNDQGVVKVSDFGLSRYVLDDEYTSSGSKFPVRWSPPEVLMYSKFSSKSDIWAFGVLMWEIYSLGKMPYERFTNSETAEHIAQGLRLYRPHLASEKVYTIMYSCWHEKADERPTFKILLSNILDVMDEE

    sequence by ProteinMPNN: (recovery: 44.906      nssr: 59.623)
    LEPWEIDPADLTYLEHLGTGPGGTVWAGLLKGKTPVAVKELKPGAFDEDALIEWLKEKMNIKHPNIVQLLGVSTGQTPILIVKEYCPKGVLLDYLRDKSRNLSPEQLLQLCLNIAKGLAYLESKNILHRDLKTGNCLVDENGVAKIADFGFIRFVRDPSARTVGSDFPYRWSPLEVLTNGNYSFASDVWSFGVTMWEIFSLGATPFAGMTNEEIIAYIKAGKTLTRPALASPAAWALAAACLAPNPADRPTFAELLAALEAILAAA

    sequence by ProRefiner + ProteinMPNN: (recovery: 56.226 nssr: 72.453)
    MGAWEINPADLTFLEHLGEGALGVVRKGLLKGKTKVAVKELKEGAFDIESLIADAKVKMNLKHENLVRLYGICTSQSPILLITEYMANGNLLDYLRDKSRNFSTEQLLQMCLDVCKAMAYLESKNELHRDLKSENCLVDENGVVKVSDYGLIRFVKDESARTVGSKFPVRWSPPEVLENNDYSFKSDVWSFGVTMWEIFSLGETPYESMSDEETAAWIKQGKTLTRPARASDEVWAILSSCLQADAEQRPTFAELLAQLEEVQKAE


### Partial sequence design
Run the following script for partial design, where the indexes of residues to design (index starting from 1 not 0) are **separated by comma**.

    python run.py PDB_CODE_or_PDB_PATH CHAIN INDEX1,INDEX2,INDEX3

Please note that there is no space betweem indexes. For example, to design the first 10 residues of chain A, run:

    python run.py 8flh A 1,2,3,4,5,6,7,8,9,10

An example output for the above command will be:

    Read 8flh chain A with length 266
    Recovery and nssr are computed based on designable residues with coordinates (10 residues).

    native sequence:
    YGSWEIDPKDLTFLKELGTGQFGVVKYGKWRGQYDVAIKMIKEGSMSEDEFIEEAKVMMNLSHEKLVQLYGVCTKQRPIFIITEYMANGCLLNYLREMRHRFQTQQLLEMCKDVCEAMEYLESKQFLHRDLAARNCLVNDQGVVKVSDFGLSRYVLDDEYTSSGSKFPVRWSPPEVLMYSKFSSKSDIWAFGVLMWEIYSLGKMPYERFTNSETAEHIAQGLRLYRPHLASEKVYTIMYSCWHEKADERPTFKILLSNILDVMDEE

    sequence by ProteinMPNN: (recovery: 40.000      nssr: 50.000)
    LEPYEIDISDLTFLKELGTGQFGVVKYGKWRGQYDVAIKMIKEGSMSEDEFIEEAKVMMNLSHEKLVQLYGVCTKQRPIFIITEYMANGCLLNYLREMRHRFQTQQLLEMCKDVCEAMEYLESKQFLHRDLAARNCLVNDQGVVKVSDFGLSRYVLDDEYTSSGSKFPVRWSPPEVLMYSKFSSKSDIWAFGVLMWEIYSLGKMPYERFTNSETAEHIAQGLRLYRPHLASEKVYTIMYSCWHEKADERPTFKILLSNILDVMDEE

    sequence by ProRefiner: (recovery: 50.000       nssr: 90.000)
    MGAWEVNPEDLTFLKELGTGQFGVVKYGKWRGQYDVAIKMIKEGSMSEDEFIEEAKVMMNLSHEKLVQLYGVCTKQRPIFIITEYMANGCLLNYLREMRHRFQTQQLLEMCKDVCEAMEYLESKQFLHRDLAARNCLVNDQGVVKVSDFGLSRYVLDDEYTSSGSKFPVRWSPPEVLMYSKFSSKSDIWAFGVLMWEIYSLGKMPYERFTNSETAEHIAQGLRLYRPHLASEKVYTIMYSCWHEKADERPTFKILLSNILDVMDEE


