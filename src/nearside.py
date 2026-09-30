from dataclasses import dataclass, field
import time
from pathlib import Path
import sys
from tqdm import tqdm
import random

import torch

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import get_device, find_longest_repeated_block
from utils.embedder import EmbeddingModel, EmbedderName
from utils.vlm import VLM, VLMName
from utils.dataset import Dataset, DatasetSource
from existing_attacks.cwa import modify_model_processor
from utils.model import get_model

"""
Defense proposed in https://arxiv.org/pdf/2410.22888v1
"""

@dataclass
class NearsideConfig:
    device: str = "cuda"
    vlm_name:VLMName = VLMName.SMOLVLM_1_2B # the target VLM to be defended
    user_query: str = "What is in the image?"
    adv_projection_threshold: float = -15#why?

class NearSideDefense:
    def __init__(self, config: NearsideConfig, victim_model=None):
        self.config = config
        self.vlm = VLM(config.vlm_name, config.device) if victim_model is None else victim_model#get_model(config.vlm_name, config.device)

        #modify_model_processor([self.vlm])#, do_print=True)#do we want this?

    def detect(self, image: torch.tensor, user_query:str = None) -> tuple[bool, float]:
        emb = self.get_embedding(image, user_query = user_query)
        adv_projection = torch.dot(emb, self.adv_direction/torch.norm(self.adv_direction, p=2))
        is_adv = adv_projection > self.config.adv_projection_threshold

        return is_adv, adv_projection

    def get_embedding(self, image: torch.tensor, user_query=None):
        #modify_model_processor([self.vlm])

        # get the hidden state deciding the first generated token
        if user_query is None:
            user_query = self.config.user_query
        test_prompt = self.vlm.get_test_prompt(user_query)
        outputs = self.vlm.forward(image, test_prompt, overwrite=True, output_attentions=True, output_hidden_states=True)
        hidden_states = torch.cat(outputs.hidden_states, dim=0)
        # print(hidden_states.shape)
        # print(outputs.image_hidden_states.shape)
        embedding = hidden_states[-1,-1,:].flatten()
        del outputs, hidden_states
        return embedding

    def set_adv_direction(self, adv_direction):
        self.adv_direction = adv_direction

def make_safe_name(s: str) -> str:
    return s.replace("/", "_")

def load_or_compute_adv_direction(ds=None, defense=None, attack_config=None, victim=2, filename=None):
    #here = (Path(__file__).parent / "tensors").resolve()
    #filename = f"{here}/" + make_safe_name(f'nearside_adv_dir_vlm_{defense.config.vlm_name.value}.pt')
    filename=f'nearside_adv_dir_vlm_{victim}.pt' if filename is None else filename

    if Path(filename).exists():
        adv_direction = torch.load(filename)
        print("loaded adv dir.")
    else:
        for sampled_idx, (img, label, target) in tqdm(enumerate(ds)):
            #modify_model_processor([defense.vlm])#, do_preprocess=True) only if NIPS
            org_img = img.clone()
            adv_image=torch.load(f'bigmodels_adversarial_data/adaptize/0/naib/adv__{sampled_idx}_{victim}_16.0_0.pt')
            adv_image = adv_image/255
            org_emb = defense.get_embedding(org_img).detach()
            adv_emb = defense.get_embedding(adv_image).detach()
            #input([org_emb, adv_emb])
            # depending on whether this is the first iteration
            try:
                adv_direction += (adv_emb - org_emb) / len(ds)
            except:
                adv_direction = (adv_emb - org_emb) / len(ds)

        torch.save(adv_direction, filename)

    return adv_direction


if __name__ == "__main__":
    random.seed(42)
    device = 'cuda'#get_device(prefer_mps=True)
    ds = Dataset(DatasetSource.NIPS_17, device)
    ds_train, ds_test = ds.split_train_test(train_ratio=0.1)

    # comment later
    ds_train = ds_train.create_susbset(10)
    ds_test = ds_test.create_susbset(15)


    config = NearsideConfig(
        vlm_name=VLMName.SMOLVLM_1_2B,
    )
    victim=2
    defense = NearSideDefense(config)

    # load or train svm classifier
    adv_direction = load_or_compute_adv_direction(ds_train, defense)
    defense.set_adv_direction(adv_direction)

    # test
    benign_detections, adv_detections = [], []
    for i, (img, label, target) in enumerate(ds_test):
        sampled_idx = i + 10
        org_img = img.clone()
        org_img.requires_grad = True
        is_adv, y_pred = defense.detect(org_img)
        benign_detections.append(is_adv.item())
        print(f"Time {time.ctime()} -> Benign Image {i} Results:- is_adv: {is_adv}, y_pred: {y_pred}")

        modify_model_processor([defense.vlm])#, do_preprocess=True)
        adv_image=torch.load(f'bigmodels_adversarial_data/adaptize/0/pip/adv__{sampled_idx}_{victim}_16.0_0.pt')
        adv_image = adv_image/255

        is_adv, y_pred = defense.detect(adv_image)

        adv_detections.append(is_adv.item())
        print(f"Time {time.ctime()} -> Adv Image {i} Results:- is_adv: {is_adv}, y_pred: {y_pred}")

    print(f"Recall: {sum(adv_detections) / len(adv_detections):.2f}")
    print(f"FPR: {sum(benign_detections) / len(benign_detections):.2f}")
