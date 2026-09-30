## The Adversarial Gait: Detecting Adversarial Attacks against Vision-Language Models via Self-Targeted Gradient Characterization (NeurIPS2026)
<img width="931" height="266" alt="image" src="https://github.com/user-attachments/assets/88d34216-9e1a-4895-ae2b-b1a9b540ef20" />


This repository contains the implementation for the core components of the paper. The code should be run from the src/ folder, and is organized as follows:

directories:
- data/, existing_attacks/, utils/: helper files required to run our implementation
- archive/, archive_coco/, archive_flickr/ data used for our evaluation on NIPS2017, MSCOCO, and Flickr, respectively.

main files:
- utils/transfer_attack_ms.py: implementation of our adaptive attacks and AdverStep
- pip.py, nearside.py, mirrorcheck.py: implementation of our baseline detectors
- attack_nips_ms.py: short script that launches adaptive attacks against AdverGait and the baselines against specific victim models on specific evaluation data. 
- pipsvm_x.joblib, nearside_adv_dir_vlm_x.pt: pre-trained components to perform detection with PIP and Nearside
- dsvq_lite.pkl, dsvq_test_lite.pkl: training and evaluation sets from the $VQA^2$ dataset, respectively.

- requirements.txt: packages required to run our code
- example_commands.txt: larger set of example commands similar to the ones presented below. 

secondary files:
- "idx" numpy files: used to track the adversarial targets for our evaluation datasets.



## EXAMPLE COMMANDS
Run adaptive attacks against AdverStep and the baselines on five NIPS2017 images:
```
#AdverStep
python attack_nips_ms.py --method gait --n_gradient_steps 1000 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 1 --d_gradient_steps 1 --ds nips --savedir outputs/

#Nearside
python attack_nips_ms.py --method ns --n_gradient_steps 1000 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 1 --d_gradient_steps 1 --ds nips --savedir outputs/

#PIP
python attack_nips_ms.py --method pip --n_gradient_steps 1000 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 1 --d_gradient_steps 1 --ds nips --savedir outputs/

#MirrorCheck
python attack_nips_ms.py --method mc --n_gradient_steps 1000 --vlm1 3 --perz 0 --max_pert 16 --npatches 0 --start 0 --stop 1 --d_gradient_steps 1 --ds nips --savedir outputs/
```
arguments:
- n_gradient_steps: maximum adaptive attack iterations
- vlm1: victim model identifier (3 is Qwen-VL2.5-3B see attack_nips_ms.py or example_commands.txt for more details)
- perz: FPR on training data (determines the detection threshold based on the training data, can only be varied for Qwen-VL-3B)
- max_pert: bound on the L_inf norm in pixels (when npatches=0) or maximum area % to be used by patch attack as a float between 0 and 1 (when npatches > 0)
- npatches: set to zero for norm-bounded attack; for larger n it is the number of adversarial patches in the image (attack budget is distributed equally across patches)
- start: index of first image to attack (between 0 and 500 for all evaluation sets)
- stop: index of last image to be attacked plus one (between 1 and 100, and > start)
- savedir: use --save to specifiy the directory where to store the resulting adversarial images and attack time statistics


## TODO
We will soon add code and example commands for:
- Additional Baselines
- Training and tuning the different detectors
