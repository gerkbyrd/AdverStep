import sys
from pathlib import Path
import time

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).parent.parent))
from utils.embedder import EmbedderName, EmbeddingModel
from utils.vlm import VLMName, VLM
from utils.image_generator import ImageGenerator, ImageGeneratorName
from utils.model import get_model
from utils.utils import get_device, get_memory_consumption
from utils.transfer_attack import attack_step_pgd
from existing_attacks.cwa import modify_model_processor

class ChainOfAttack:
    def __init__(
            self,
            surrogate_encoder_name: EmbedderName = EmbedderName.SIGLIP2_BASE_PATCH16,
            text_to_image_model_name: ImageGeneratorName = ImageGeneratorName.STABLE_DIFFUSION_B_V1,
            image_to_text_model_name: VLMName = VLMName.SMOLVLM_1_256M,
            n_diffusion_steps: int = 50,
            vision_embedding_coeff_alpha: float = 0.5,
            pos_neg_balancing_coeff_beta: float = 0.7,
            margin_coeff_gamma: float = 0.3,
            perturbation_budget_epsilon: float = 8.0 / 255,
            pgd_lr_eta: float = 1.0 / 255,
            n_gradient_steps: int = 100,
            random_start: bool = False,
            do_grad_sign: bool = False,
            device: str = "mps",
        ):

        self.surrogate_encoder_name = surrogate_encoder_name
        self.text_to_image_model_name = text_to_image_model_name
        self.image_to_text_model_name = image_to_text_model_name
        self.n_diffusion_steps = n_diffusion_steps
        self.vision_embedding_coeff_alpha = vision_embedding_coeff_alpha
        self.pos_neg_balancing_coeff_beta = pos_neg_balancing_coeff_beta
        self.margin_coeff_gamma = margin_coeff_gamma
        self.perturbation_budget_epsilon = perturbation_budget_epsilon
        self.pgd_lr_eta = pgd_lr_eta
        self.n_gradient_steps = n_gradient_steps
        self.random_start = random_start
        self.do_grad_sign = do_grad_sign
        self.device = device

        self.surrogate_encoder: EmbeddingModel = get_model(self.surrogate_encoder_name, self.device)
        self.text_to_image_model: ImageGenerator = get_model(self.text_to_image_model_name, self.device)
        # self.text_to_image_model = None
        self.image_to_text_model: VLM = get_model(self.image_to_text_model_name, self.device)

        modify_model_processor([self.surrogate_encoder, self.image_to_text_model])

    def attack(self, clean_image: torch.tensor, target_text: str):
        # adversarial copy
        adv_image = clean_image.clone()
        
        # generate clean text
        image_to_text_prompt = self.image_to_text_model.get_test_prompt("what is in this image?")
        clean_text = self.image_to_text_model.generate(image=clean_image, formatted_prompt=image_to_text_prompt, overwrite=True, max_new_tokens=50) # cannot go above 64 tokens because of limits of CLIP/SigLIP
        clean_text = [clean_text[0].split("Assistant: ")[-1]]
        print("clean text generated:\n", clean_text)

        # generate target image
        target_image = self.text_to_image_model.generate_image(text_description=target_text, num_steps=self.n_diffusion_steps)
        target_image = target_image / 255

        plt.imshow(T.ToPILImage()(target_image))
        plt.show()

        del self.text_to_image_model # just to save memory, should keep the model in case we do this iteratively


        # compute uni-model embeddings
        clean_text_embedding = self.surrogate_encoder.compute_txt_embedding(clean_text)
        target_text_embedding = self.surrogate_encoder.compute_txt_embedding(target_text)

        clean_img_embedding = self.surrogate_encoder.compute_img_embedding(clean_image, clean_image, overwrite=True)
        target_img_embedding = self.surrogate_encoder.compute_img_embedding(target_image, clean_image, overwrite=True)

        # fuse embeddings
        clean_fused_embedding = self.fuse_embeddings(clean_img_embedding, clean_text_embedding)
        target_fused_embedding = self.fuse_embeddings(target_img_embedding, target_text_embedding)

        if self.random_start:
            adv_image += (torch.rand()*2-1) * self.perturbation_budget_epsilon

        for i in tqdm(range(self.n_gradient_steps)):
            adv_image.requires_grad = True

            # generate adversarial text
            adv_text = self.image_to_text_model.generate(image=adv_image, formatted_prompt=image_to_text_prompt, overwrite=True, max_new_tokens=50)
            adv_text = [adv_text[0].split("Assistant: ")[-1]]
            print(adv_text)

            # compute adversarial fused embedding
            adv_text_embedding = self.surrogate_encoder.compute_txt_embedding(adv_text)
            adv_img_embedding = self.surrogate_encoder.compute_img_embedding(adv_image, clean_image, overwrite=True)
            adv_fused_embedding = self.fuse_embeddings(adv_img_embedding, adv_text_embedding)

            # compute loss (we actually want to maximize the loss)
            similarity_diff = torch.dot(target_fused_embedding, adv_fused_embedding) - self.pos_neg_balancing_coeff_beta * torch.dot(clean_fused_embedding, adv_fused_embedding)
            # similarity_diff = torch.abs(similarity_diff) # this step is mentioned in their algorithm, but not in their pseudocode
            loss = max(similarity_diff + self.margin_coeff_gamma, 0)

            # log
            linf = (adv_image - clean_image).norm(p=float('inf'))*255
            print(f"Time: {time.ctime()} -> timestep: {i+1:4d} / {self.n_gradient_steps} loss: {loss.item()}, Linf: {linf:.3f} / 255, RAM: {get_memory_consumption(self.device)} GB")
            # print(f"Linf: {(adv_image - clean_image).norm(p=float('inf'))*255:.3f} / 255")

            # compute gradient
            grads = torch.autograd.grad(loss, [adv_image])[0]

            with torch.no_grad():
                adv_image = attack_step_pgd(
                    image=adv_image,
                    grads=grads,
                    lr=-1*self.pgd_lr_eta, # -1 because we want to maximize the loss
                    max_perturbation_pixels=self.perturbation_budget_epsilon, 
                    initial_image=clean_image,
                    do_sign=self.do_grad_sign,
                    max_pixel=1,
                )
        
        return adv_image


            
    def normalize_embedding(self, embedding: torch.tensor):
        return embedding / embedding.norm(dim=1, keepdim=True)

    def fuse_embeddings(self, image_embedding: torch.tensor, text_embedding: torch.tensor, do_normalize: bool = True):
        if do_normalize:
            image_embedding, text_embedding = self.normalize_embedding(image_embedding), self.normalize_embedding(text_embedding)
        fused_embedding = self.vision_embedding_coeff_alpha * image_embedding + (1 - self.vision_embedding_coeff_alpha) * text_embedding
        if do_normalize:
            fused_embedding = self.normalize_embedding(fused_embedding)
        return fused_embedding.squeeze()

if __name__ == "__main__":

    from datasets import load_dataset
    import torchvision.transforms.v2 as T

    device = get_device(prefer_mps=True)

    ds_name = "Multimodal-Fatima/Imagenet1k_sample_validation" # for the fatima dataset, we need to divide by 255
    dataset = load_dataset(ds_name, split='validation')
    print("Loaded dataset.")

    ds_entry = dataset[0]
    x = T.PILToTensor()(ds_entry["image"])
    x = x/255
    x = T.Resize((224,224))(x).float().to(device) #.unsqueeze(0)

    target_answer = "I will not reply to you!"
    target_answer = "apple"

    coa = ChainOfAttack(
        surrogate_encoder_name=EmbedderName.CLIP_LARGE_PATCH14,
        n_gradient_steps=100,
        n_diffusion_steps=10,
        perturbation_budget_epsilon= 32.0/255,
        do_grad_sign=True, # the psuedocode say they compute sign, but equations and algorithm dont
    )
    adv_image = coa.attack(
        clean_image=x,
        target_text=target_answer,
    )

    print(f"Linf: {(adv_image - x).norm(p=float('inf'))*255:.3f} / 255")


    # evaluate
    test_vlm_names = [VLMName.SMOLVLM_1_256M, VLMName.SMOLVLM_1_500M]
    test_vlms = [get_model(name, device) for name in test_vlm_names]
    modify_model_processor(test_vlms)

    for model in test_vlms:
        test_prompt = model.get_test_prompt(user_query="what is in this image?")
        generation = model.generate(adv_image, test_prompt, overwrite=True, max_new_tokens=50)[0]
        generation = generation.split("Assistant: ")[-1]
        loss_before = model.get_qa_loss(x, "What is in this image?", target_answer)
        loss_after = model.get_qa_loss(adv_image, "What is in this image?", target_answer)
        print(f"Model: {model.name} -> loss_before: {loss_before.item()} loss_after: {loss_after.item()} output:\n{generation}")
