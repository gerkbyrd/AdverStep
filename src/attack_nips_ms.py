from utils.vlm import VLM, VLMName

from utils.utils import get_device, get_memory_consumption, plot_images

from datasets import load_dataset
from torchvision import transforms as T
from pathlib import Path
import os
import torch
from PIL import Image

import csv
import random
from utils.dataset import Dataset, DatasetSource
from torch.utils.data import DataLoader
from enum import Enum, auto
from pip import *
from mirrorcheck import *
from nearside import *
from utils.diffpure import ConfigYami, ConfigCus, RevGuidedDiffusion, default_args_cfg#this should be able to replace all those imports above
import string

import numpy as np
import pickle
import json
from tqdm import tqdm
import argparse
import copy

from utils.transfer_attack_multistep import AttackConfig, launch_attack
from utils.transfer_attack_multistep import det_thresh_dict as thdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gradient_steps",default=1000,type=int,help="attack steps for attack stage")# (cyclic opt)")
    parser.add_argument("--d_gradient_steps",default=10,type=int,help="attack steps for detection")
    parser.add_argument("--naive",action='store_true',help="non-adaptive attack!")
    parser.add_argument("--start",default=0,type=int,help="strating idx to consider")
    parser.add_argument("--stop",default=1,type=int,help="stopping idx to consider")
    parser.add_argument("--npatches",default=0,type=int,help="adv. patches (0 means norm-bounded attack)")
    parser.add_argument('--max_pert', type=float, default=8.0, help='adversarial perturbation budget (l_inf norm for norm-bounded, total_area% for patches)')

    parser.add_argument('--method', default="gait", type=str,help="defense")
    parser.add_argument('--ds', default="nips", type=str,help="dataset")
    parser.add_argument('--ablate', default=None, type=str,help="features to drop (feat1_feat2_ etc.)")
    parser.add_argument('--dspz', default=None, type=str,help="eval dataset to tune FPR to")
    parser.add_argument('--eval', action='store_true',help="do evaluation only")
    parser.add_argument('--ceval', action='store_true',help="eval on clean images")

    parser.add_argument("--download_dsvq",action='store_true',help="get dsvq if the 1000/500 .pkl version is not available...")


    parser.add_argument("--vlm1",default=3,type=int,help="VICTIM VLM (2: SMOLVLM-1B, 3: QWENVL-3B, 5: INTERNVL3-2B)")


    parser.add_argument('--lr', type=float, default=1.0, help='attacker learning rate')
    parser.add_argument('--glr', type=float, default=8.0, help='defender learning rate (x255)')
    parser.add_argument('--topk', type=float, default=0.05, help='maskable region')


    parser.add_argument("--perz",default=0,type=int,help="Training Data FPR%")
    parser.add_argument("--sigsharp",default=10,type=int,help="Sigmoid sharpness for BPDA")
    parser.add_argument("--cw",action='store_true',help="do CW attack instead of PGD")
    parser.add_argument("--orth",action='store_true',help="do Orthogonal PGD")
    parser.add_argument('--adam', action='store_true',help="use Adam optimizer")
    parser.add_argument("--stubborn",action='store_true',help="do not break upon success (naive attack)")

    #
    #PIP
    parser.add_argument("--occ",action='store_true',help="one-class SVM")

    #MirrorCheck
    parser.add_argument("--mc_dsteps",default=20,type=int,help="MirrorCheck diffusion steps")
    parser.add_argument("--mc_gsteps",default=1,type=int,help="MirrorCheck sampled images")

    parser.add_argument('--savedir', default="outputs/", type=str,help="path to save adversarial images and computation cost results")

    parser.add_argument('--strparam', default="nutnet_results/", type=str,help="path to data to avoid")

    args = parser.parse_args()
    relative_home = ""#./cv_transfer/src/"




    cls_file = Path(os.path.abspath('')).parents[0] / "data" / "imagenet1000_clsidx_to_labels.txt"
    relative_home = "./"#cv_transfer/src/"
    device  = get_device()#prefer_mps=True)
    #input(device)
    with open(cls_file, "r") as f:
        idx2label = eval(f.read())

    VLMS = [
        VLMName.SMOLVLM_1_256M,#OG (300M)
        VLMName.SMOLVLM_1_500M,#500M
        VLMName.SMOLVLM_1_2B,#2B
        VLMName.QWEN_2p5_VL_3B,#4B
        VLMName.LLAVA_ONEVISION_0p5B,#0.9B
        VLMName.INTERNVL_3_2B,
        #VLMName.INTERNVL_2p5_2B,
    ]

    model_name_vlm = VLMS[args.vlm1] if args.vlm1!=-1 else VLMS[0]

    method=args.method
    occ=args.occ
    caseno=args.vlm1
    refname = VLMS[caseno]
    refmod=VLM(refname, 'cuda')
    pipname = f"pipsvm_occ_{caseno}.joblib" if occ else f"pipsvm_{caseno}.joblib"
    nsname=f'nearside_adv_dir_vlm_{caseno}.pt'

    lr=args.lr

    p_defense, m_defense, n_defense, purifier = None, None, None, None
    if method=='mc':
        config = MirrorCheckConfig(
            image_to_text_name = refname,
            user_query = "",#does not matter because we send victim model text output as text description
            n_diffusion_steps=args.mc_dsteps,
            device=device,
        )
        m_defense = MirrorCheck(config, victim_model=refmod)

    elif method=='pip':
        config = PIPConfig(
            vlm_name=refname,
            occ=args.occ
        )
        p_defense = PIPDefense(config, victim_model=refmod)
        clf = load_or_train_svm(None, defense=p_defense, attack_config=None, file=pipname)
        p_defense.set_svm(clf)

    elif method=='ns':
        config = NearsideConfig(
            vlm_name=refname,#Name.SMOLVLM_1_256M,
        )
        n_defense = NearSideDefense(config, victim_model=refmod)
        adv_direction = load_or_compute_adv_direction(None, defense=n_defense, filename=nsname)
        n_defense.set_adv_direction(adv_direction)

    elif method == 'diffpure':
        args_difp, cfg_difp = default_args_cfg()
        purifier = RevGuidedDiffusion(args_difp,cfg_difp)
        
    print_every=50
    perz =  args.perz
    #speed...
    mag = args.max_pert
    gradsteps = args.n_gradient_steps
    npatches=args.npatches
    patchst = '_patch' if npatches else ''

    attack_config = AttackConfig(
        model_name_clf="",
        model_name_vlm=refname,
        n_gradient_steps=gradsteps,
        d_gradient_steps=args.d_gradient_steps,
        lr=lr,#64/255,#255*(5e-4),
        lambda_clf=0,
        lambda_vlm=1,
        d_lambda_clf=0,
        d_lambda_vlm=1,
        user_query="not set",#user_query,
        target_vlm_answer="not set",#target_vlm_answer,
        target_clf_idx=0,#target_clf_idx,
        max_perturbation=mag,#for patches this means 5% of total area... #16.0/255
        d_max_perturbation=mag,
    )

    #gc.collect()
    #torch.cuda.empty_cache()
    print("MEMORY CONS. NOW")
    print(get_memory_consumption('cuda'))

    # important for choosing the same target labels each time
    random.seed(42)
    if args.ds == "nips":
        ds = Dataset(DatasetSource.NIPS_17)
        # create train and test splits
        #ds_train, ds_test = ds.split_train_test(train_ratio=0.1)
        #yets = [i for i in range(1000)]
        destiny = np.load(relative_home + "destiny_idx_nips.npy")
        yets = [i for i in range(500)]
    elif args.ds == "flickr":
        ds = Dataset(DatasetSource.FLICKR)
        destiny = np.load(relative_home + "destiny_idx_flickr.npy")
        yets = [i for i in range(500)]
    elif args.ds == "coco":
        ds = Dataset(DatasetSource.MSCOCO)
        destiny = np.load(relative_home + "destiny_idx_mscoco.npy")
        yets = [i for i in range(500)]
    elif args.ds == "dsvq":
        if args.download_dsvq:
            ds = load_dataset("lmms-lab/VQAv2", split = "validation")#~220K
            destiny = np.load(relative_home + "training_idx.npy")
            destiny_tgt = np.load(relative_home + "destiny_idx_adv.npy")
        else:
            with open(f"{args.ds}_lite.pkl", "rb") as f:
                ds = pickle.load(f)
        yets = [i for i in range(1000)]
    elif args.ds == "dsvq_test":
        #ds = load_dataset("lmms-lab/VQAv2", split = "test")#~447.8K
        if args.download_dsvq:
            ds = load_dataset("lmms-lab/VQAv2", split = "validation")#~220K
            destiny = np.load(relative_home + "test_idx.npy")
            destiny_tgt = np.load(relative_home + "test_idx_adv.npy")
        else:
            with open(f"{args.ds}_lite.pkl", "rb") as f:
                ds = pickle.load(f)
        yets = [i for i in range(1000)]


    captioning_query = 'Please carefully observe the image and come up with a caption for the image.' # Common query to all captioning datasets
    vlm_vanilla = refmod#VLM(model_name=name, device=device)
    vlmbol = []
    stepoton = args.d_gradient_steps - 1
    locusts = thdb[args.topk][refname]
    innerd=False
    getouttheway=[]
    if args.eval:
        dets=[]
    for sampled_idx in yets:#, (img, lbl, tgt) in enumerate(dataloader):
        if sampled_idx not in range(args.start, args.stop):
            continue
        innerd = True
        if not innerd and not args.cw:# and not args.eval:
            print("DAYYYYE")
            print(sampled_idx)
            continue
        #print("GO-HO-HO")
        #print(img.shape, lbl, tgt)
        if args.ds in ["dsvq", "dsvq_test"]:#"coco", "flickr", "nips", "fat", "nocaps", "txtcaps"]:

            #if enough diskpace to load entire VQA2...
            if args.download_dsvq:
                sampled_image = ds[destiny[sampled_idx]]["image"]
                user_query = ds[destiny[sampled_idx]]["question"]
                adv_label = ds[destiny_tgt[sampled_idx]]["answers"][0]["answer"]
            else:
                sampled_image = ds[sampled_idx][0]
                user_query = ds[sampled_idx][1]#ds[destiny[sampled_idx]]["question"]
                adv_label = ds[sampled_idx][2]#ds[destiny_tgt[sampled_idx]]["answers"][0]["answer"]

            image_tensor = T.PILToTensor()(sampled_image)
            attack_config.target_clf_idx=0
            attack_config.target_vlm_answer=adv_label
            #print([user_query, adv_label])
            #getouttheway.append([sampled_image, user_query, adv_label])
            #continue
        elif args.ds in ["fat"]:
            pass

        elif args.ds in ["nips", "coco", "flickr"]:
            img, lbl, tgt = ds[int(destiny[sampled_idx])]
            if args.ds == "nips":
                attack_config.target_clf_idx = tgt
                attack_config.target_vlm_answer = "No, it is a " + idx2label[tgt].split(",")[0].strip() + "."
                user_query = "Is this a " + idx2label[lbl].split(",")[0].strip() + "?"
            else:
                attack_config.target_clf_idx = 0
                attack_config.target_vlm_answer = tgt
                user_query = captioning_query
        print(f"TARGET: {attack_config.target_vlm_answer}")
        attack_config.n_gradient_steps = gradsteps

        if args.ds in ["nips", "coco", "flickr"]:
            image_tensor = (img*255).to(torch.uint8)
            sampled_image = T.ToPILImage()(image_tensor)

        #image_tensor = T.PILToTensor()(sampled_image)


        if image_tensor.shape[0]==1:
            image_tensor = T.PILToTensor()(Image.merge("RGB", (sampled_image, sampled_image, sampled_image)))
            image_tensor = T.Resize((224,224))(image_tensor).float()

        image_tensor = T.Resize((224,224))(image_tensor).float().squeeze(0)
        attack_config.user_query = user_query
        #input("AYE")
        if args.eval and args.n_gradient_steps > 1:#method=="naib":
            #print("LOAD WHAT IS RIGHT")
            #input(patchst)
            #image_tensor = torch.load(f'bigmodels_adversarial_data/adaptize/golden/{args.ds}/0/naib_ms_{args.d_gradient_steps}/' + f'adv_{patchst}_{sampled_idx}_{caseno}_{mag}_{npatches}.pt')
            try:
                image_tensor = torch.load(f'bigmodels_adversarial_data/adaptize/{args.ds}/0/naib_ms_{args.d_gradient_steps}_8.0/' + f'adv_{patchst}_esc_obj_{sampled_idx}_{caseno}_{mag}_{npatches}.pt')
            except:
                image_tensor = torch.load(f'bigmodels_adversarial_data/adaptize/{args.ds}/0/naib_ms_{args.d_gradient_steps}_8.0/' + f'adv_{patchst}_obj_{sampled_idx}_{caseno}_{mag}_{npatches}.pt')
        elif False:#args.eval:
            method_e = "naib"
            savedir = f'bigmodels_adversarial_data/adaptize/{args.ds}/{perz}/{method_e}_ms_{args.d_gradient_steps}/'
            nam = "esc_obj"
            try:
                image_tensor = torch.load(savedir + f'adv_{patchst}_{nam}_{sampled_idx}_{caseno}_{mag}_{npatches}.pt')
                print("B4 PURIFIED:")
                print(vlm_vanilla.generate_e2e(image_tensor.clone().detach(), user_query, temperature=0)[0])
                print("AFTER PURIFIED:")
            except:
                dets.append(1)
        #input("try")
        if args.eval and not args.ceval:
            image_tensor = torch.load(f'bigmodels_adversarial_data/adaptize/{args.ds}/0/naib_ms_{args.d_gradient_steps}_8.0/' + f'adv_{patchst}_obj_{sampled_idx}_{caseno}_{mag}_{npatches}.pt')#torch.load(f'bigmodels_adversarial_data/adaptize/golden/{args.ds}/0/naib_ms_{args.d_gradient_steps}/' + f'adv_{patchst}_{sampled_idx}_{caseno}_{mag}_{npatches}.pt')
        if args.eval and args.n_gradient_steps == 1 and args.method=="gait":
            curout=vlm_vanilla.generate_e2e(image_tensor.clone().detach(), user_query, temperature=0)[0]
            suxbeg= attack_config.target_vlm_answer.lower().translate(str.maketrans('', '', string.punctuation)) == curout.lower().translate(str.maketrans('', '', string.punctuation))
            attack_config.target_vlm_answer=curout

        bonanza, itertimes = launch_attack(
                                image=image_tensor.cuda(),
                                vlmi=vlm_vanilla,
                                config=attack_config,
                                print_every=print_every,
                                device=device,
                                morevlm=vlmbol,
                                untargeted=False,#for untargeted only!
                                patch_attack=npatches>0,
                                npatches=npatches,
                                pipdef=p_defense,
                                mcdef=m_defense,
                                nsdef=n_defense,
                                dpdef=purifier,
                                nmc=args.mc_gsteps,
                                naib=method=='naib',
                                #gait=gait,
                                #refsims = torch.tensor(),
                                topk=args.topk,
                                perz=perz,
                                ablate=args.ablate,
                                cw=args.cw,
                                gait_lr=args.glr/255,
                                stubborn=args.stubborn,
                                dspz=args.dspz,
                                adam=args.adam,
                                sigsharp=args.sigsharp,
                                orth=args.orth,
                            )
        if args.eval:
            if args.n_gradient_steps > 1:
                det_out = bonanza[6]==None#(method!="gait" and bonanza)
            elif args.method=="gait":
                det_out = [bonanza, suxbeg]
            else:
                det_out=bonanza
            dets.append(det_out)
            continue
        #input("safety")
        if args.ablate is not None:
            savedir = f'{args.savedir}/{args.ds}/{perz}/{method}_{args.ablate}_ms_{args.d_gradient_steps}/'
        elif args.cw:
            savedir = f'{args.savedir}/{args.ds}/{perz}/{method}_cw_ms_{args.d_gradient_steps}/'
        elif args.adam:
            savedir = f'{args.savedir}/{args.ds}/{perz}/{method}_adam_ms_{args.d_gradient_steps}/'
        elif args.dspz is not None:
            savedir = f'{args.savedir}/{args.ds}/{perz}_{args.dspz}/{method}_ms_{args.d_gradient_steps}_{args.glr}/'
        elif args.sigsharp!=10:
            savedir = f'{args.savedir}/{args.ds}/{perz}/{method}_ms_{args.d_gradient_steps}_{args.glr}_sig{args.sigsharp}/'
        elif args.orth:
            savedir = f'{args.savedir}/{args.ds}/{perz}/{method}_ms_{args.d_gradient_steps}_{args.glr}_opgd/'
        else:
            savedir = f'{args.savedir}/{args.ds}/{perz}/{method}_ms_{args.d_gradient_steps}_{args.glr}/'
        os.makedirs(relative_home + savedir, exist_ok=True)

        for ni, nam in enumerate(['obj', 'all', 'stealth_1', 'stealth_2', 'stealth_all',
                              'esc_obj', 'esc_obj_big', 'esc_stealth', 'esc_stealth_big']):
            print(len(bonanza))
            if bonanza[ni+1]!=None:
                torch.save(bonanza[ni+1], (relative_home + savedir + f'adv_{patchst}_{nam}_{sampled_idx}_{caseno}_{mag}_{npatches}.pt'))
                if nam =="esc_obj" and args.d_gradient_steps!=4 and method == "gait":
                    print("TAKE A LOOK")
                    attack_config.target_vlm_answer = vlm_vanilla.generate_e2e(bonanza[ni+1], user_query, temperature=0)[0]
                    attack_config.n_gradient_steps = 1
                    bonanza_s, _ = launch_attack(
                                            image=bonanza[ni+1].cuda(),
                                            vlmi=vlm_vanilla,
                                            config=attack_config,
                                            print_every=print_every,
                                            device=device,
                                            morevlm=vlmbol,
                                            untargeted=False,#for untargeted only!
                                            patch_attack=npatches>0,
                                            npatches=npatches,
                                            pipdef=p_defense,
                                            mcdef=m_defense,
                                            nsdef=n_defense,
                                            dpdef=purifier,
                                            nmc=args.mc_gsteps,
                                            naib=method=='naib',
                                            #gait=gait,
                                            #refsims = torch.tensor(),
                                            topk=args.topk,
                                            perz=perz,

                                        )
                    feats = bonanza_s#[0]
                    #input(feats)
                    print(feats)
                    detection = 0

                    detection = detection > 0
                    print(f"DETECTION IS THUS: {detection}")
        else:
            meinevs= np.array(itertimes)
            with open(relative_home + savedir + f'adv_{patchst}_{sampled_idx}_{caseno}_{mag}_{npatches}_itertimes.npy', 'wb') as f:
                np.save(f, meinevs)

    if args.eval:
        print(f'{args.ds}/{perz}/{method}/' + f'adv_{patchst}_{caseno}_{mag}_{npatches}')
        print(dets)
        if args.n_gradient_steps > 1:
            print(sum(dets))
            print(f"EVAL RESULT: {sum(dets)/100}")
