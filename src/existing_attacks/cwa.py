"""
Code from https://github.com/huanranchen/AdversarialAttacks/blob/dbac7e7e32844f440876b57e6c0d3bbb95040a80/attacks/AdversarialInput/AdversarialInputBase.py
"""

import torch
from typing import Callable, List, Iterable
from torch import nn
import random
from torchvision import transforms
import numpy as np
from scipy import stats as st
import os
from pathlib import Path
import sys


import torch
from abc import abstractmethod
from typing import List
from torch import Tensor
from math import ceil

sys.path.append(str(Path(__file__).parent.parent))
from utils.vlm import VLM

"""
In case we want the images to be [0,255]:
1. updtae the clamp method's max_value
2. self.epsilon = epsilon *255
3. self.inner_step_size *= 255
   self.step_size *= 255
   self.reverse_step_size *= 255
4. ksi *= 255
5. comment x = x/255
6. comment model.processor.do_rescale=False
this does not give exact same results but very similar
"""

"""
NOTE: For VLMs, do_ensemble_logits_not_loss might not work if tokenizers are different for different VLMs
"""

def clamp(x: torch.tensor, min_value=0, max_value=255):
    return torch.clamp(x, min=min_value, max=max_value)


def inplace_clamp(x: torch.tensor, min_value: float = 0, max_value: float = 1):
    return x.clamp_(min=min_value, max=max_value)

def cosine_similarity(x: list):
    '''
    input a list of tensor with same shape. return the mean cosine_similarity
    '''
    x = torch.stack(x, dim=0)
    N = x.shape[0]
    x = x.reshape(N, -1)

    norm = torch.norm(x, p=2, dim=1)
    x /= norm.reshape(-1, 1)  # N, D
    similarity = x @ x.T  # N, N
    # matrix_heatmap(similarity.cpu().numpy())
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=0).to(torch.bool)
    similarity = similarity[mask]
    return torch.mean(similarity).item()

class AdversarialInputAttacker():
    def __init__(self, model: List[torch.nn.Module],
                 epsilon=16 / 255,
                 norm='Linf'):
        assert norm in ['Linf', 'L2']
        self.norm = norm
        self.epsilon = epsilon
        self.models = [m.model for m in model]
        self.model_objects = model
        self.init()
        # self.model_distribute()
        self.device = torch.device('cuda')
        self.n = len(self.models)

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.attack(*args, **kwargs)

    def model_distribute(self):
        '''
        make each model on one gpu
        :return:
        '''
        num_gpus = torch.cuda.device_count()
        models_each_gpu = ceil(len(self.models) / num_gpus)
        for i, model in enumerate(self.models):
            model.to(torch.device(f'cuda:{num_gpus - 1 - i // models_each_gpu}'))
            model.device = torch.device(f'cuda:{num_gpus - 1 - i // models_each_gpu}')

    def init(self):
        # set the model parameters requires_grad is False
        for model in self.models:
            model.requires_grad_(False)
            model.eval()

    def to(self, device: torch.device):
        for model in self.models:
            model.to(device)
            model.device = device
        self.device = device

    def clamp(self, x: Tensor, ori_x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.norm == 'Linf':
            x = torch.clamp(x, min=ori_x - self.epsilon, max=ori_x + self.epsilon)
        elif self.norm == 'L2':
            difference = x - ori_x
            distance = torch.norm(difference.view(B, -1), p=2, dim=1)
            mask = distance > self.epsilon
            if torch.sum(mask) > 0:
                difference[mask] = difference[mask] / distance[mask].view(torch.sum(mask), 1, 1, 1) * self.epsilon
                x = ori_x + difference
        x = torch.clamp(x, min=0, max=1)
        return x


"""
Code from https://github.com/huanranchen/AdversarialAttacks/blob/dbac7e7e32844f440876b57e6c0d3bbb95040a80/attacks/AdversarialInput/CommonWeakness.py
"""
class MI_CommonWeakness(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = False,
                 step_size: float = 16 / 255 / 5,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 mu=1,
                 outer_optimizer=None,
                 reverse_step_size=16 / 255 / 15,
                 inner_step_size: float = 250,
                 DI=False,
                 TI=False,
                 use_custom_model_objects=True,
                 do_ensemble_logits_not_loss=True,
                 *args,
                 **kwargs
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.mu = mu
        self.outer_optimizer = outer_optimizer
        self.reverse_step_size = reverse_step_size
        super(MI_CommonWeakness, self).__init__(model, *args, **kwargs)
        self.inner_step_size = inner_step_size
        self.DI = DI
        self.TI = TI
        self.use_custom_model_objects = use_custom_model_objects
        self.do_ensemble_logits_not_loss = do_ensemble_logits_not_loss
        if DI:
            self.aug_policy = transforms.Compose([
                transforms.RandomCrop((int(224 * 0.9), int(224 * 0.9)), padding=224 - int(224 * 0.9)),
            ])
        else:
            self.aug_policy = nn.Identity()
        if TI:
            self.ti = self.gkern().to(self.device)
            self.ti.requires_grad_(False)

    def perturb(self, x):
        x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
        x = clamp(x)
        return x

    def attack(self, x, y, ):
        N = x.shape[0]
        original_x = x.clone()
        inner_momentum = torch.zeros_like(x)
        self.outer_momentum = torch.zeros_like(x)
        if self.random_start:
            x = self.perturb(x)

        for step_i in range(self.total_step):
            # --------------------------------------------------------------------------------#
            # first step
            self.begin_attack(x.clone().detach())

            logit = 0
            self.target_class_idx = y[0]
            loss = 0
            #input(x)
            x.requires_grad = True
            #input(self.do_ensemble_logits_not_loss)
            for model in self.model_objects:
                if self.use_custom_model_objects:
                    #x=model.processor(x.to("cpu"), return_tensors="pt").to(self.device)['pixel_values']

                    log = model.forward_logits(x)#.to(model.device))
                else:
                    log = model.model(x.to(model.device)).logits

                if self.do_ensemble_logits_not_loss:
                    logit += log
                else:
                    #log.requires_grad=True
                    #b=torch.sum(log**2)
                    #b.backward()

                    loss += model.compute_clf_loss(log, self.target_class_idx)
                    #ggrad = torch.autograd.grad(loss, x)[0]
                    #loss.backward()
                    #print(x.grad)
                    #print(ggrad)
                    #input(loss)

            if self.do_ensemble_logits_not_loss:
                loss = self.model_objects[0].compute_clf_loss(logit, self.target_class_idx)
            print(step_i, "loss:", loss)
            loss.backward()
            grad = x.grad
            if self.TI:
                grad = self.ti(grad)
            x.requires_grad = False
            if self.targerted_attack:
                x += self.reverse_step_size * grad.sign()
            else:
                x -= self.reverse_step_size * grad.sign()
                # x -= self.reverse_step_size * grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(N, 1, 1, 1)
            x = clamp(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            # --------------------------------------------------------------------------------#
            # second step
            x.grad = None
            # self.begin_attack(x.clone().detach())
            for model in self.model_objects:
                x.requires_grad = True
                aug_x = self.aug_policy(x)
                if self.use_custom_model_objects:
                    logits = model.forward_logits(aug_x.to(model.device))
                    loss = model.compute_clf_loss(logits, self.target_class_idx)
                else:
                    model = model.model
                    loss = self.criterion(model(aug_x.to(model.device)).logits, y.to(model.device))
                loss.backward()
                grad = x.grad
                self.grad_record.append(grad)
                x.requires_grad = False
                # update
                if self.TI:
                    grad = self.ti(grad)
                if self.targerted_attack:
                    inner_momentum = self.mu * inner_momentum - grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                else:
                    inner_momentum = self.mu * inner_momentum + grad / torch.norm(grad.reshape(N, -1), p=2, dim=1).view(
                        N, 1, 1, 1)
                    x += self.inner_step_size * inner_momentum
                x = clamp(x)
                x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            x = self.end_attack(x)
            x = clamp(x, original_x - self.epsilon, original_x + self.epsilon)

        return x

    @torch.no_grad()
    def begin_attack(self, origin: torch.tensor):
        self.original = origin
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self, now: torch.tensor, ksi=16 / 255 / 5):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = now
        if self.outer_optimizer is None:
            fake_grad = (patch - self.original)
            self.outer_momentum = self.mu * self.outer_momentum + fake_grad / torch.norm(fake_grad, p=1)
            patch.mul_(0)
            patch.add_(self.original)
            patch.add_(ksi * self.outer_momentum.sign())
            # patch.add_(ksi * fake_grad)
        else:
            fake_grad = - ksi * (patch - self.original)
            self.outer_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original)
            patch.grad = fake_grad
            self.outer_optimizer.step()
        patch = clamp(patch)
        grad_similarity = cosine_similarity(self.grad_record)
        del self.grad_record
        del self.original
        return patch

    @staticmethod
    def gkern(kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = torch.tensor(kernel, dtype=torch.float)
        conv = nn.Conv2d(3, 3, kernel_size=kernlen, stride=1, padding=kernlen // 2, bias=False, groups=3)
        kernel = kernel.repeat(3, 1, 1).view(3, 1, kernlen, kernlen)
        conv.weight.data = kernel
        return conv


def evaluate_loss(model, image, target_label_idx, use_custom_model_objects: bool=False, is_targeted_attack: bool=False):
    if use_custom_model_objects:
        logits = model.forward_logits(image)
    else:
        logits= model.model(image).logits
    loss = model.compute_clf_loss(logits, torch.tensor(target_label_idx).to(model.device))

    if False:#type(model) == ImageClassifier:
        topk = logits.topk(k=10).indices
        attack_success = (topk[0] == target_label_idx) if is_targeted_attack else (topk[0] != target_label_idx)
    else: attack_success = None
    return loss.item(), attack_success

def modify_model_processor(models: list):
    for model in models:
        if hasattr(model.processor, "image_processor"):
            print(f"Disabling scaling for model: {model.name}")
            # this is a VLM or embedding model
            # print(model.processor.image_processor.__dict__)
            # model.processor.image_processor.do_resize=False
            # model.processor.image_processor.do_normalize=False
            model.processor.image_processor.do_rescale=False
            # model.processor.image_processor.do_center_crop=False
        else:
            # this is a classifier
            # print(model.processor._valid_kwargs_names)
            model.processor.do_resize=False
            model.processor.do_normalize=False
            model.processor.do_rescale=False
            model.processor.do_center_crop=False # does not help


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.vlm import VLMName
    from utils.embedder import EmbedderName
    from datasets import load_dataset
    from utils.model import get_model
    from utils.utils import get_device, get_memory_consumption
    import torchvision.transforms.v2 as T
    import matplotlib.pyplot as plt
    from data.nips17 import get_NIPS17_loader

    use_custom_model_objects = True
    do_ensemble_logits_not_loss = False
    is_targeted_attack = True
    target_label_idx = 850

    device  = get_device(prefer_mps=True)

    #model_names = [ClassifierName.VIT_L_P16_224, ClassifierName.RESNET_50]
    #test_model_names = [ClassifierName.VIT_B_P16_224]

    model_names = [VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M]
    test_model_names = [VLMName.SMOLVLM_1_2B]


    ds_name = "Multimodal-Fatima/Imagenet1k_sample_validation" # for the fatima dataset, we need to divide by 255
    dataset = load_dataset(ds_name, split='validation')
    print("Loaded dataset.")
    models = [get_model(model_name, device) for model_name in model_names]
    test_models = models + [get_model(tmn, device) for tmn in test_model_names]

    cls_file = Path(os.path.abspath(__file__)).parent.parent.parent / "data" / "imagenet1000_clsidx_to_labels.txt"
    with open(cls_file, "r") as f:
        idx2label = eval(f.read())


    ds_entry = dataset[0]
    x = T.PILToTensor()(ds_entry["image"])
    x = x/255
    x = T.Resize((224,224))(x).float().to(device).unsqueeze(0)

    #x = T.Resize((224,224))(x).float()
    #if x.shape[0]==1:
        #x = T.PILToTensor()(Image.merge("RGB", (x, x, x)))
        #x = T.Resize((224,224))(x).float()

    if not is_targeted_attack:
        target_label_idx = ds_entry["label"]

    for model in test_models:
        if type(model) == VLM:
            model.set_default_prompt_info(
                user_query="What is in this image?",
                target_label_str=idx2label[target_label_idx].split(",")[0].strip(),
            )

    y = torch.tensor([target_label_idx]).to(device)


    """
    Image has to be between 0 and 1 for the attack to work
    Image procesors rescale (/255) by default, make sure you disable it if image is already in [0,1]
    """
    modify_model_processor(test_models)

    attack = MI_CommonWeakness(
        model=models,
        targeted_attack=is_targeted_attack,
        use_custom_model_objects=use_custom_model_objects,
        do_ensemble_logits_not_loss=do_ensemble_logits_not_loss,
        total_step=10,
        # mu=0,
    )

    #print([m.device for m in models])
    #print([m.device for m in attack.model_objects])
    #input()

    original = x.clone()
    attacked = attack.attack(x, y)
    print(attacked)
    input()
    original_pil = T.ToPILImage()(original.cpu().squeeze())
    attacked_pil = T.ToPILImage()(attacked.cpu().squeeze())
    silly_adv = original + 0.1*torch.randn_like(original)
    # plt.imshow(original_pil)
    # plt.show()
    # plt.imshow(attacked_pil)
    # plt.show()

    for i, m in enumerate(test_models):
        loss_org, attack_success_org = evaluate_loss(m, original, target_label_idx, use_custom_model_objects, is_targeted_attack)
        loss_adv, attack_success_adv = evaluate_loss(m, attacked, target_label_idx, use_custom_model_objects, is_targeted_attack)
        loss_sil, attack_success_sil = evaluate_loss(m, silly_adv, target_label_idx, use_custom_model_objects, is_targeted_attack)
        print(f"Model: {m.name.value}, Loss before: {loss_org}, Loss after: {loss_adv}, Loss silly: {loss_sil}, \n\tSuccess before: {attack_success_org}, Success after: {attack_success_adv}, Success silly: {attack_success_sil}")
