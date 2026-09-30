## The Adversarial Gait: Detecting Adversarial Attacks against Vision-Language Models via Self-Targeted Gradient Characterization
<img width="931" height="266" alt="image" src="https://github.com/user-attachments/assets/88d34216-9e1a-4895-ae2b-b1a9b540ef20" />


This repository contains the implementation for the core components of the paper. The code is organized as follows:

directories:
- archive/, data/, existing_attacks/, utils/: helper files required to run our implementation
- out_script/: stores adaptive attack results

main files:
- utils/transfer_attack.py: algorithm used for both our adaptive attacks and AdverGait
- pip.py, nearside.py, mirrorcheck.py: implementation of our baseline detectors
- attack_nips.py: short scripts that launches adaptive attacks against AdverGait and the baselines on the NIPS2017 dataset
- pipsvm_x.joblib, nearside_adv_dir_vlm_x.pt: pre-trained components to perform detection with PIP and Nearside



## EXAMPLE COMMANDS
Run adaptive attacks against AdverGait and the baselines on five NIPS2017 images:
```
#AdverGait
python attack_nips.py --method gait --n_gradient_steps 200 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 5

#PIP
python attack_nips.py --method pip --n_gradient_steps 200 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 5

#Nearside
python attack_nips.py --method ns --n_gradient_steps 200 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 5

#MirrorCheck
python attack_nips.py --method mc --n_gradient_steps 200 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 5
```
arguments:
- n_gradient_steps: maximum adaptive attack iterations
- vlm1: victim model identifier (3 is Qwen-VL2.5-3B see attack_nips.py for more details)
- perz: FPR on training data (determines the detection threshold based on the training data, can only be varied for Qwen-VL-3)
- max_pert: bound on the L_inf norm in pixels (when npatches=0) or maximum area % to be used by patch attack as a float between 0 and 1 (when npatches > 0)
- npatches: set to zero for norm-bounded attack; for larger n it is the number of adversarial patches in the image (attack budget is distributed equally across patches)
- start: index of first image to attack in NIPS2017 (between 0 and 99)
- stop: index of last image to be attacked plus one (between 1 and 100, and > start)
- save: use --save to store the resulting adversarial images and attack time statistics in the out_script/ folder
