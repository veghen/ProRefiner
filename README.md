# An Entropy-based Refining Strategy for Inverse Protein Folding with Global Graph Attention

This is a demo code for paper An Entropy-based Refining Strategy for Inverse Protein Folding with Global Graph Attention. You can also run the demo online through [Colab](https://colab.research.google.com/drive/1a6VW-BB0twEwL65sE_dUAM42wdSm6RZp?usp=sharing) or [Code Ocean](https://codeocean.com/capsule/7d2d57dd-96ec-48aa-9e3f-8f4a0b7d0150/) for easier environment setup. 

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

    python run.py PDB_CODE CHAIN

For example:
    
    python run.py 8flh A

The program will download the PDB file of the given PDB code, and run sequence design on the specified chain (only single chain design is supported). Here is an example output of the above script:

    Design 265 residues from 8flh chain A (ignore residues without coordinates)

    native sequence:
    YGSWEIDPKDLTFLKELGTGQFGVVKYGKWRGQYDVAIKMIKEGSMSEDEFIEEAKVMMNLSHEKLVQLYGVCTKQRPIFIITEYMANGCLLNYLREMRHRFQTQQLLEMCKDVCEAMEYLESKQFLHRDLAARNCLVNDQGVVKVSDFGLSRYVLDDEYTSSGSKFPVRWSPPEVLMYSKFSSKSDIWAFGVLMWEIYSLGKMPYERFTNSETAEHIAQGLRLYRPHLASEKVYTIMYSCWHEKADERPTFKILLSNILDVMDE

    sequence by ProteinMPNN: (recovery: 43.774	nssr: 58.113)
    LKPYEIDPKDLTIEEHLGTGGGGTVWKGLYKGKTPVAIKELKPGRFDEDALIAYMEEKMNIKHPNIVQLFGISSSGTPILKVKEYCAKGGLLAYLRDASRNLTPAQLLQLCIDIAKGMAYLESKNILHRDLKTGNCLVDENDVAKVADYGGILFVKDPEARTVGSKFPVRWSPLEVLENGDYSFASDVWSFGVTMYEIFSRGATPFAGMTDEEIRAYIAAGGTLTRPPLASPAMWAIADSCLARDPSDRPTFAEILAALEAEAAA

    sequence by ProRefiner + ProteinMPNN: (recovery: 55.472	nssr: 71.321)
    MGEWEINPKDLTFLEHLGTGALGVVYKGLYKGKKKVAVKELKEGAFDIESLIADSKVRMNLKHENLVQLYGICTSSSPILLVVEYMANGNLLDYLRDKSRNFSTEQLLQMCLDVAKAMAYLESKNELHRDLKSENCLVDENGVVKVSDYGLIRFVKNEEARTVGSKFPVRWSPPEVLENNDYSFKSDVWSFGVTMWEIFSLGATPFEDMSDEETAEWIRAGKTLTRPALASDAVWAILSSCLQRDASKRPTFAELLKQLREVQKK

Note that when invalid chain code is provided, the program will return an error. For example, the output of script `python run.py 8flh F` will be

    Chain F not found in 8flh (chains: ['A'])

### Partial sequence design
Run the following script for partial design, where the indexes of residues to design (index starting from 1 not 0) are **separated by comma**.

    python run.py PDB_CODE CHAIN INDEX1,INDEX2,INDEX3

Please note that there is no space betweem indexes. For example, to design the first 10 residues of chain A, run:

    python run.py 8flh A 1,2,3,4,5,6,7,8,9,10

The program will only output the sequences for designable residues. An example output for the above command will be:

    Design 10 residues from 8flh chain A (ignore residues without coordinates)

    native sequence:
    YGSWEIDPKD

    sequence by ProteinMPNN: (recovery: 40.000	nssr: 50.000)
    LEPYEIDISD

    sequence by ProRefiner: (recovery: 50.000	nssr: 90.000)
    MGAWEVNPED

