from dataclasses import dataclass, field
import time
from pathlib import Path
import sys
from tqdm import tqdm


import torch
from sklearn import svm
import joblib


#sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import get_device, find_longest_repeated_block
from utils.embedder import EmbeddingModel, EmbedderName
from utils.vlm import VLM, VLMName, VLMS
from utils.dataset import Dataset, DatasetSource
from torch.utils.data import DataLoader

import torchvision.transforms as T
from PIL import Image
import argparse
import numpy as np
import random
#from existing_attacks.cwa import modify_model_processor
#from utils.transfer_attack import AttackConfig, ClassificationTarget, launch_attack

def modify_model_processor(models: list, do_preprocess: bool = False, do_print: bool = False):
    """
    If image is already scales [0,1], we dont need to do it again
    """
    action = "Disabling" if not do_preprocess else "Enabling"
    for model in models:
        if hasattr(model.processor, "image_processor"):
            if do_print: print(f"{action} scaling for model: {model.name}")
            # this is a VLM or embedding model
            # print(model.processor.image_processor.__dict__)
            # model.processor.image_processor.do_resize=do_preprocess
            # model.processor.image_processor.do_normalize=do_preprocess
            model.processor.image_processor.do_rescale=do_preprocess
            # model.processor.image_processor.do_center_crop=do_preprocess
        else:
            # this is a classifier
            # print(model.processor._valid_kwargs_names)
            model.processor.do_resize=do_preprocess
            model.processor.do_normalize=do_preprocess
            model.processor.do_rescale=do_preprocess
            model.processor.do_center_crop=do_preprocess # does not help

@dataclass
class PIPConfig:
    device: str = "cuda"#"mps"
    vlm_name:VLMName = VLMName.SMOLVLM_1_256M # the target VLM to be defended
    user_query: str = "Is there a clock?"
    svm_threshold: float = 0.5
    occ: bool = False


class PIPDefense:
    def __init__(self, config: PIPConfig, victim_model=None):
        self.config = config
        self.vlm = VLM(config.vlm_name, config.device, attn_implementation="eager") if victim_model is None else victim_model# we need to be eager to get the attention
        if victim_model is None:
            modify_model_processor([self.vlm], do_print=True)
        self.mod=victim_model is not None

    def detect(self, image: torch.tensor, user_query: str = None, score=False):
        user_query = self.config.user_query if user_query is None else user_query
        attentions = self.get_attention(image, user_query)
        return self.feed_attention_to_svm(attentions, score=score)

    def set_svm(self, svm):
        self.svm = svm

    def feed_attention_to_svm(self, attention, score=False):
        if score:
            out = self.svm.decision_function([attention.flatten().tolist()])
            if self.config.occ:
                out = -out
            return out
        y_pred =  self.svm.predict([attention.flatten().tolist()])
        if self.config. occ:
            is_adv = y_pred < self.config.svm_threshold
        else:
            is_adv = y_pred > self.config.svm_threshold
        return is_adv, y_pred



    def get_attention(self, image, do_max=True) -> torch.tensor:
        modify_model_processor([self.vlm])
        # identify where the image tokens start and end (they are the most common token)
        test_prompt = self.vlm.get_test_prompt(user_query=self.config.user_query)
        inputs = self.vlm.create_vlm_inputs(image, test_prompt, overwrite=True)
        start, end, img_token_id = find_longest_repeated_block(inputs.input_ids[0].tolist())

        # get the attentions deciding the first generated token
        outputs = self.vlm.forward(image, test_prompt, overwrite=True, output_attentions=True, output_hidden_states=True)
        attn = torch.cat(outputs.attentions, dim=0) # concatenate all layers -> [layers, heads, output_token, input_token]
        attn = attn[:,:,-1,start:end+1] # [layers, heads, input_token]

        # they take the maximum head in the paper
        if do_max:
            attn = torch.max(attn, dim=1).values # [layers, input_token]
        if self.mod:
            modify_model_processor([self.vlm], do_preprocess=True)
        return attn

def make_safe_name(s: str) -> str:
    return s.replace("/", "_")

def load_or_train_svm(ds: Dataset, defense: PIPDefense, attack_config=None, file=None, victim=0):# AttackConfig):
    #here = (Path(__file__).parent / "svms").resolve()
    #filename = f"{here}/" + make_safe_name(f'svm_model_vlm_{defense.config.vlm_name.value}.joblib')

    if file is not None:#Path(filename).exists():
        with open(file, 'rb') as f:
            clf = joblib.load(f)
    else:
        X, y = [], []
        for sampled_idx, (img, lbl, tgt) in enumerate(ds):
            if sampled_idx not in range(80,100):
                continue
            #for fatima...
            """
            sampled_image = ds.images[sampled_idx]#/255
            image_tensor = T.PILToTensor()(sampled_image)
            image_tensor = T.Resize((224,224))(image_tensor).float()
            if image_tensor.shape[0]==1:
                #continue
                image_tensor = T.PILToTensor()(Image.merge("RGB", (sampled_image, sampled_image, sampled_image)))
                image_tensor = T.Resize((224,224))(image_tensor).float()
            """
            org_img, label, target = img,lbl,tgt#ds[i]
            #org_img=image_tensor/255
            adv_image = torch.load(relative_home + 'adversarial_data/adv_best_{}_0_{}_nips.pt'.format(sampled_idx, victim)).detach()/255#torch.load('cowboylivinglife/adv_{}_0_-1_50.pt'.format(sampled_idx))/255#adv_image/255
            org_attn = defense.get_attention(org_img).flatten().tolist()
            adv_attn = defense.get_attention(adv_image).flatten().tolist()
            X.extend([org_attn, adv_attn])
            y.extend([0, 1])

        clf = svm.SVC()
        clf.fit(X, y)
        # Save the model to a file
        with open("pipsvm_zero_{}.joblib".format(victim), 'wb') as f:
            joblib.dump(clf, f)
    return clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action='store_true',help="train an SVM")
    parser.add_argument("--start",default=0,type=int,help="strating idx to consider")
    parser.add_argument("--stop",default=1,type=int,help="stopping idx to consider")

    parser.add_argument("--vlm1",default=1,type=int,help="victim model")

    args = parser.parse_args()
    relative_home = "./cv_transfer/src/"

    device = 'cuda'#get_device(prefer_mps=True)
    #ds = Dataset(DatasetSource.IMAGENET_FATIMA, device)
    random.seed(42)
    ds = Dataset(DatasetSource.NIPS_17)
    ds_train, ds_test = ds.split_train_test(train_ratio=0.5)
    dataloader = DataLoader(ds_train, batch_size=1, shuffle=False)

    #ds_train=ds_test=ds

    config = PIPConfig(
        vlm_name=VLMS[args.vlm1],
    )
    defense = PIPDefense(config)
    """
    attack_config = AttackConfig(
        model_names=[defense.config.vlm_name],
        n_gradient_steps=20,
        lr=255*(1e-3),
        lambdas=[1],
        user_query=defense.config.user_query,
        classification_target=None,
        max_perturbation=8.0/255,
    )
    """
    # load or train svm classifier
    if args.train:
        clf = load_or_train_svm(ds_train, defense, attack_config=None, victim=args.vlm1)
        print("training done")
    else:
        clf = load_or_train_svm(ds_train, defense, attack_config=None, file="pipsvm_{}.joblib".format(args.vlm1))

        defense.set_svm(clf)

        # test
        benign_detections, adv_detections = [], []
        scores=[]
        times=[]
        for sampled_idx, (img, lbl, tgt) in enumerate(dataloader):
            if sampled_idx not in range(args.start, args.stop):
                continue
            i = sampled_idx
            image, label, target = img,lbl,tgt#ds[i]
            #fatima:::
            """
            sampled_image=ds.images[sampled_idx]#.clone()
            image_tensor = T.PILToTensor()(sampled_image)
            image_tensor = T.Resize((224,224))(image_tensor).float()
            if image_tensor.shape[0]==1:
                #continue
                image_tensor = T.PILToTensor()(Image.merge("RGB", (sampled_image, sampled_image, sampled_image)))
                image_tensor = T.Resize((224,224))(image_tensor).float()
            org_img = image_tensor/255#img.clone()
            """
            org_img=image.squeeze(0)
            #is_adv, y_pred = defense.detect(org_img, score=True)
            start=time.process_time()
            score=defense.detect(org_img, score=True)
            end=time.process_time()
            times.append(end-start)
            scores.append(similarities)

            #benign_detections.append(is_adv.item())
            i = sampled_idx
            #print(f"Time {time.ctime()} -> Benign Image {i} Results:- is_adv: {is_adv}, y_pred: {y_pred}")
            """
            modify_model_processor([defense.vlm], do_preprocess=True)
            attack_config.classification_target = ClassificationTarget(target, ds.idx2label)
            adv_image = launch_attack(
                image=img*255,
                models=[defense.vlm],
                config=attack_config,
                print_every=25,
                device=device,
            )
            """
            adv_image = torch.load(relative_home + 'adversarial_data/adv_best_{}_0_{}_nips.pt'.format(sampled_idx, args.vlm1)).detach()/255#torch.load('cowboylivinglife/adv_{}_0_-1_50.pt'.format(sampled_idx))/255#adv_image/255

            #is_adv, y_pred = defense.detect(adv_image)
            start=time.process_time()
            score=defense.detect(adv_image, score=True)
            end=time.process_time()
            times.append(end-start)
            scores.append(similarities)
            with open(relative_home + 'pip_output_{}_{}.npy'.format(sampled_idx, args.vlm1), 'wb') as f:
                np.save(f, np.array(scores))
            with open(relative_home + 'pip_time_{}_{}.npy'.format(sampled_idx, args.vlm1), 'wb') as f:
                np.save(f, np.array(times))
            #adv_detections.append(is_adv.item())
            #print(f"Time {time.ctime()} -> Adv Image {i} Results:- is_adv: {is_adv}, y_pred: {y_pred}")

        #print(f"Recall: {sum(adv_detections) / len(adv_detections):.2f}")
        #print(f"FPR: {sum(benign_detections) / len(benign_detections):.2f}")
