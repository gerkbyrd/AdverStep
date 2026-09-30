from dataclasses import dataclass, field
import time
from pathlib import Path
import sys

import torch

#sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import get_device
from utils.embedder import EmbeddingModel, EmbedderName
from utils.vlm import VLM, VLMName
from utils.image_generator import ImageGenerator, ImageGeneratorName
from utils.dataset import Dataset, DatasetSource
from torch.utils.data import DataLoader

#import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
import argparse
import numpy as np
import random


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
class MirrorCheckConfig:
    detection_threshold: float = 0.1
    image_to_text_name: VLMName = VLMName.SMOLVLM_1_256M
    text_to_image_name: ImageGeneratorName = ImageGeneratorName.STABLE_DIFFUSION_B_V1
    image_encoder_names: list[EmbedderName] = field(default_factory=lambda: [EmbedderName.CLIP_BASE_PATCH16])#, EmbedderName.CLIP_LARGE_PATCH14])
    user_query: str = "What is in the image?"
    max_generated_tokens: int = 30
    n_diffusion_steps: int = 20
    device: str = "cuda"#"mps"

class MirrorCheck:
    def __init__(self, config: MirrorCheckConfig, victim_model=None, sim_only=False):
        self.device = config.device
        self.config = config
        if not sim_only:
            self.i2t_model = VLM(config.image_to_text_name, config.device) if victim_model is None else victim_model
            self.t2i_model = ImageGenerator(config.text_to_image_name, config.device)
        self.image_encoders = [EmbeddingModel(e, config.device) for e in config.image_encoder_names]
        rest = [self.i2t_model] if (not sim_only and victim_model is None) else []
        modify_model_processor(self.image_encoders + rest, do_print=True)

    def detect(self, image: torch.tensor, user_query:str = None) -> tuple[bool, float]:
        modify_model_processor(self.image_encoders)# + [self.i2t_model])
        user_query = self.config.user_query if user_query is None else user_query
        while True:
            text_description = self.i2t_model.generate_e2e(image*255, user_query, max_new_tokens=self.config.max_generated_tokens, temperature=0)[0]
            if False:#"I'm sorry" in text_description or "I apologize" in text_description:
                pass
            else:
                break

        print(f"Text description: {text_description}")
        gen_image = self.t2i_model.generate_image(text_description, self.config.n_diffusion_steps)
        similarities = [enc.compute_similarity_two_images(image, gen_image) for enc in self.image_encoders]
        avg_sim = sum(similarities) / len(similarities)
        is_adv = avg_sim < self.config.detection_threshold
        #modify_model_processor([self.i2t_model], do_preprocess=True)
        return is_adv, similarities, gen_image

    def generate(self, text_description: str) -> float:
        #modify_model_processor(self.image_encoders + [self.i2t_model])
        print(f"Text description: {text_description}")
        gen_image = self.t2i_model.generate_image(text_description, self.config.n_diffusion_steps)
        return gen_image

    def similarity(self, image1: torch.tensor, image2: torch.tensor) -> float:
        modify_model_processor(self.image_encoders)# + [self.i2t_model])
        similarities = [enc.compute_similarity_two_images(image1, image2) for enc in self.image_encoders]
        avg_sim = sum(similarities) / len(similarities)
        return avg_sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",default=0,type=int,help="strating idx to consider")
    parser.add_argument("--stop",default=1,type=int,help="stopping idx to consider")

    parser.add_argument("--vlm1",default=1,type=int,help="victim model")
    args = parser.parse_args()

    relative_home = "./cv_transfer/src/"
    device = 'cuda'#get_device(prefer_mps=True)
    #ds = Dataset(DatasetSource.IMAGENET_FATIMA, device)
    #ds_train=ds_test=ds

    random.seed(42)
    ds = Dataset(DatasetSource.NIPS_17)
    ds_train, ds_test = ds.split_train_test(train_ratio=0.5)
    dataloader = DataLoader(ds_train, batch_size=1, shuffle=False)
    #device = get_device(prefer_mps=True)
    #ds = Dataset(DatasetSource., device)
    print("Dataset Loaded!")
    config = MirrorCheckConfig(
        user_query = "What is in the below image?",
        n_diffusion_steps=10,
        device=device,
    )
    defense = MirrorCheck(config)
    print("Defense initialized!")


    do_show_images = False
    benign_detections, adv_detections = [], []
    scores=[]
    times=[]
    for sampled_idx, (img, lbl, tgt) in enumerate(dataloader):
        if sampled_idx not in range(args.start, args.stop):
            continue
        i = sampled_idx
        image, label, target = img,lbl,tgt#ds[i]
        #fatima way:
        """
        #sampled_image=ds.images[sampled_idx]#.clone()
        #lab=ds.labels[sampled_idx]
        image_tensor = T.PILToTensor()(sampled_image)
        image_tensor = T.Resize((224,224))(image_tensor).float()
        if image_tensor.shape[0]==1:
            #continue
            image_tensor = T.PILToTensor()(Image.merge("RGB", (sampled_image, sampled_image, sampled_image)))
            image_tensor = T.Resize((224,224))(image_tensor).float()
        image = image_tensor/255#img.clone()
        """
        # test defense on bening image
        #print(f"Testing benign image {i}. Benign label: {lab}")#"{ds.idx2label_fn(label)}")
        start=time.process_time()
        is_adv, similarities, gen_image = defense.detect(image)
        end=time.process_time()
        times.append(end-start)
        scores.append(similarities)

        benign_detections.append(is_adv)
        if do_show_images:
            plt.imshow(T.ToPILImage()(gen_image))
            plt.show()
        #print(f"Time {time.ctime()} -> Benign Image {i} Results:- is_adv: {is_adv}, similarities: {similarities}")

        # test defense on adversarial image
        print(f"Testing adversarial image {i}")#". Converting class '{ds.idx2label_fn(label)}' to '{ds.idx2label_fn(target)}'")
        """
        modify_model_processor([defense.i2t_model], do_preprocess=True)
        attack_config.classification_target = ClassificationTarget(target, ds.idx2label)
        adv_image = launch_attack(
            image=image*255,
            models=[defense.i2t_model],
            config=attack_config,
            print_every=25,
            device=device,
        )
        adv_image = adv_image/255
        """
        adv_image = torch.load(relative_home + 'adversarial_data/adv_best_{}_0_{}_nips.pt'.format(sampled_idx, args.vlm1)).detach()#torch.load('cowboylivinglife/adv_{}_0_-1_50.pt'.format(sampled_idx))/255#

        start=time.process_time()
        is_adv, similarities, gen_image = defense.detect(adv_image)
        end=time.process_time()
        times.append(end-start)
        scores.append(similarities)

        adv_detections.append(is_adv)
        #print(f"Time {time.ctime()} -> Adv Image {i} Results:- is_adv: {is_adv}, similarities: {similarities}")
        with open(relative_home + 'mirrorcheck_output_{}_{}.npy'.format(sampled_idx, args.vlm1), 'wb') as f:
            np.save(f, np.array(scores))
        with open(relative_home + 'mirrorcheck_time_{}_{}.npy'.format(sampled_idx, args.vlm1), 'wb') as f:
            np.save(f, np.array(times))

    print(f"Recall: {sum(adv_detections) / len(adv_detections):.2f}")
    print(f"FPR: {(sum(benign_detections) / len(benign_detections)):.2f}")
