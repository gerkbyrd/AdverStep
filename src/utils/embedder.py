from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from strenum import StrEnum
from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

from utils.image_utils import process_image
from utils.utils import plot_images
from tqdm import tqdm


class EmbedderName(StrEnum):
    CLIP_BASE_PATCH16 = "openai/clip-vit-base-patch16"
    CLIP_LARGE_PATCH14 = "openai/clip-vit-large-patch14"
    SIGLIP2_BASE_PATCH16 = "google/siglip2-base-patch16-224"
    SIGLIP2_LARGE_PATCH16 = "google/siglip2-large-patch16-256"
    CLIP_BASE_PATCH32 = "openai/clip-vit-base-patch32"
    CLIP_LAION_14 = "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    JINA_CLIP_2 = "jinaai/jina-clip-v2"
    E5_V = "royokong/e5-v"
    COLSMOL_500M = "vidore/colSmol-500M"
    COLSMOL_256M = "vidore/colSmol-256M"
    SMOLVLM_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"  # just to test if it runs?
    COLPALI = "vidore/colpali-v1.3"
    QWEN2_GME_2B = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    QWEN2_GME_7B = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"


CLIP_LIKE_MODELS = [
    EmbedderName.CLIP_BASE_PATCH16,
    EmbedderName.CLIP_LARGE_PATCH14,
    EmbedderName.SIGLIP2_BASE_PATCH16,
    EmbedderName.SIGLIP2_LARGE_PATCH16,
    EmbedderName.CLIP_BASE_PATCH32,
    EmbedderName.CLIP_LAION_14,
]

COLSMOL_MODELS = [
    EmbedderName.COLSMOL_500M,
    EmbedderName.COLSMOL_256M,
    EmbedderName.SMOLVLM_256M  # NOTE: we can use any VLM as if it was a colpali model
]

COLPALI_MODELS = COLSMOL_MODELS + [
    EmbedderName.COLPALI,
]


class EmbeddingLoss(StrEnum):
    MSE = "mse"
    COS = "cos"
    MAXSIM = "maxsim"
    AVGSIM = "avgsim"
    SOFTMAXSIM = "softmaxsim"
    COS_AVGEMB = "cos_avgemb"


COLPALI_LOSSES = [
    EmbeddingLoss.MAXSIM,
    EmbeddingLoss.AVGSIM,
    EmbeddingLoss.SOFTMAXSIM,
    EmbeddingLoss.COS_AVGEMB,
]
NON_COLPALI_LOSSES = [
    EmbeddingLoss.COS,
    EmbeddingLoss.MSE
]

QWEN_GME_MODELS = [
    EmbedderName.QWEN2_GME_2B,
    EmbedderName.QWEN2_GME_7B,
]

ALL_EMBEDDERS = [
    EmbedderName.CLIP_BASE_PATCH16,
    EmbedderName.CLIP_LARGE_PATCH14,
    EmbedderName.SIGLIP2_BASE_PATCH16,
    EmbedderName.SIGLIP2_LARGE_PATCH16,
    EmbedderName.JINA_CLIP_2,
    EmbedderName.COLSMOL_500M,
    EmbedderName.COLSMOL_256M,
    EmbedderName.COLPALI,
    EmbedderName.QWEN2_GME_2B,
    EmbedderName.QWEN2_GME_7B,
    EmbedderName.CLIP_BASE_PATCH32,
    EmbedderName.CLIP_LAION_14,
]


def is_loss_compatible(model_name_emb: EmbedderName, loss: EmbeddingLoss) -> bool:
    if (model_name_emb in COLPALI_MODELS and loss not in COLPALI_LOSSES) or (
            model_name_emb not in COLPALI_MODELS and loss in COLPALI_LOSSES):
        return False
    return True


# candidate models
MODEL_NAMES = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14",
    "google/siglip2-base-patch16-224",
    "jinaai/jina-clip-v2",
    "vidore/colSmol-256M",  # maybe not worth trying since it requires installing colpali
    # larger models: test later
    "royokong/e5-v",
    "nomic-ai/nomic-embed-vision-v1.5",
    # multimodal retrieval requires using this in conjunction with "nomic-ai/nomic-embed-text-v1.5"
    "vidore/colpali-v1.3",
]


class EmbeddingModel:
    def __init__(self, model_name: EmbedderName, device):
        self.name = model_name
        self.device = device

        if model_name == EmbedderName.JINA_CLIP_2:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else "auto",
                trust_remote_code=True).to(device)
            self.processor = None
            self.tokenizer = None

        if model_name in CLIP_LIKE_MODELS:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else "auto").to(device)
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if model_name == EmbedderName.COLPALI:
            from colpali_engine.models import ColPali, ColPaliProcessor

            self.model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else torch.bfloat16).to(device)
            self.processor = ColPaliProcessor.from_pretrained(model_name)
            self.tokenizer = None

        if model_name in COLSMOL_MODELS:
            from colpali_engine.models import ColIdefics3, ColIdefics3Processor
            self.model = ColIdefics3.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else torch.bfloat16).to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = ColIdefics3Processor.from_pretrained(model_name)
            self.processor.image_processor.do_image_splitting = False

        elif model_name in QWEN_GME_MODELS:
            self.model = AutoModelForImageTextToText.from_pretrained(model_name).to(device).eval()
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self.tokenizer = None
            self.instruction = "You are a helpful AI model."


        # self.model.requires_grad_(False)
        self.model.eval()

    def get_classification_loss(self, image: torch.tensor, config, text: str=None, reduce=True, overwrite=True, image_query=""):
        if text is None: text = config.classification_target.target_class_str
        if isinstance(text, str): text = [text]
        mock_image = image.clone()
        image_embedding = self.compute_img_embedding(image, mock_image, overwrite=overwrite, image_query=image_query)
        if "siglip" in self.name.lower():
            target_embedding = torch.cat(tuple(self.compute_txt_embedding(t) for t in text), dim=0)
        else:
            target_embedding = self.compute_txt_embedding(text)
        if reduce:
            loss = self.compute_embedding_loss(image_embedding, target_embedding)
        else:
            loss = torch.tensor([self.compute_embedding_loss(image_embedding, te.unsqueeze(0)) for te in target_embedding])
        return loss

    # @torch.no_grad()
    def classify(self, image, idx2label):
        labels = [idx2label[i].split(", ")[0].strip() for i in range(len(idx2label))]
        losses = self.get_classification_loss(image, config=None, text=labels, reduce=False)
        return int(torch.argmin(losses)), losses.topk(k=5, largest=False)

    @torch.no_grad()
    def compare_embeddings(self, image, user_query, overwrite=False, loss_type: EmbeddingLoss = EmbeddingLoss.MSE, image_query=""):
        self.model.eval()
        if type(user_query) == str: user_query = [user_query]

        user_query_embedding = self.compute_txt_embedding(user_query)
        image_embedding = self.compute_img_embedding(image, image, overwrite, image_query)

        return self.compute_embedding_loss(image_embedding, user_query_embedding, loss_type).item()

    def compute_txt_embedding(self, user_query):

        if self.name == EmbedderName.JINA_CLIP_2:
            return torch.tensor(self.model.encode_text(user_query)).to(self.device).type(self.model.dtype)

        if self.name in CLIP_LIKE_MODELS:
            user_query_embedding = self.model.get_text_features(
                **self.tokenizer(user_query, return_tensors="pt", truncation=True, padding=True).to(
                    self.device))  # it had [0].detach()
            return user_query_embedding

        if self.name == EmbedderName.COLPALI or self.name in COLSMOL_MODELS:
            batch_queries = self.processor.process_queries(user_query).to(self.device)
            user_query_embedding = self.model(**batch_queries)
            return user_query_embedding

        if self.name in QWEN_GME_MODELS:
            assert isinstance(user_query, list)
            msg = [f"<|im_start|>system\n{self.instruction}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>" for q in user_query]
            inputs = self.processor(text=msg, return_tensors="pt", padding=True, padding_side="left", truncation=True).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = last_hidden_state[:, -1].contiguous()
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

        raise ValueError(f"Not supported model {self.name}!")

    def compute_img_embedding(self, image, mock_image: torch.tensor, overwrite=False, image_query=""):
        self.model.eval()

        if self.name == EmbedderName.JINA_CLIP_2:
            if isinstance(image, list):
                image = torch.cat(tuple([T.PILToTensor()(img.resize((512, 512))).unsqueeze(0) for img in image]))
            else:
                image = image.unsqueeze(0)
            embeddings = self.model.get_image_features(image.to(self.device))
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

        if self.name in CLIP_LIKE_MODELS:
            if overwrite:
                image_input_emb = self.processor(images=[mock_image.to("cpu")], return_tensors='pt').to(self.device)
                image_ppd_emb = process_image(image, self)
                image_input_emb['pixel_values'][0] = image_ppd_emb
            else:
                if not isinstance(image, list): image = [image]
                image_input_emb = self.processor(images=image, return_tensors='pt').to(self.device)

            image_embedding = self.model.get_image_features(**image_input_emb)
            return image_embedding


        if self.name == EmbedderName.COLPALI or self.name in COLSMOL_MODELS:
            if overwrite:
                image_input_emb = self.processor.process_images([T.ToPILImage()(mock_image)]).to(self.device)
                # image_input_emb = self.processor.process_images([mock_image]).to(self.device)
                image_ppd_emb = process_image(image, self)
                image_input_emb['pixel_values'][0] = image_ppd_emb
            else:
                if not isinstance(image, list): image = [image]
                image_input_emb = self.processor.process_images(image).to(self.device)

            image_embedding = self.model(**image_input_emb)
            return image_embedding

        if self.name in QWEN_GME_MODELS:
            if isinstance(image, list):
                image = torch.cat([T.Resize((512, 512))(T.PILToTensor()(im)).unsqueeze(0) for im in image], dim=0)
            n_image = 1 if len(image.shape) == 3 else image.shape[0]
            image_text = f"{image_query}\n<|vision_start|><|image_pad|><|vision_end|>"
            msg = [f"<|im_start|>system\n{self.instruction}<|im_end|>\n<|im_start|>user\n{image_text}<|im_end|>\n<|im_start|>assistant\n<|endoftext|>" for _ in range(n_image)]
            inputs = self.processor(text=msg, images=image, return_tensors="pt", padding=True, padding_side="left", truncation=True).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            embeddings = last_hidden_state[:, -1]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

        raise ValueError(f"Not supported model {self.name}!")

    def compute_similarity_two_images(self, image1: torch.tensor, image2: torch.tensor) -> float:
        emb1 = self.compute_img_embedding(image1, image1, overwrite=True)
        emb2 = self.compute_img_embedding(image2, image2, overwrite=True)
        similarity = -self.compute_embedding_loss(emb1, emb2) + 1#the loss itself is more of a "disimilarity"
        return similarity


    def compute_embedding_loss(self, image_embedding, text_embedding):
        if (self.name in COLPALI_MODELS or self.name == EmbedderName.COLPALI):
            # my version of the scoring function (allowing different losses)
            return -1 * score_multi_vector_modified(text_embedding, image_embedding, device=self.device,
                                                    loss=EmbeddingLoss.MAXSIM).mean()
        else:
            # cosine similarity
            return 1 - torch.nn.CosineSimilarity()(image_embedding, text_embedding).mean()

    def compute_learnable_sanity_contrastive_loss(
            self,
            image_ben: torch.tensor,
            image_adv: torch.tensor,
            exemplar_ben_emb: torch.tensor,
            exemplar_adv_emb: torch.tensor,
            image_benna: torch.tensor = None, # a benign image that will not be attacked (just to include a cross-image loss term)
        ):
        """
        The function takes as input both the benign and adversarial images, both after being perturbed with the learnable noise

        """
        embeddings_ben = self.compute_img_embedding(image_ben, image_ben.clone().to("cpu"), overwrite=True)
        embeddings_adv = self.compute_img_embedding(image_adv, image_adv.clone().to("cpu"), overwrite=True)

        loss_ben_ben = self.compute_embedding_loss(embeddings_ben, exemplar_ben_emb)
        loss_ben_adv = self.compute_embedding_loss(embeddings_ben, exemplar_adv_emb)
        loss_adv_ben = self.compute_embedding_loss(embeddings_adv, exemplar_ben_emb)
        loss_adv_adv = self.compute_embedding_loss(embeddings_adv, exemplar_adv_emb)

        total_loss = loss_ben_ben + loss_adv_adv - loss_ben_adv - loss_adv_ben

        if image_benna != None:
            embeddings_benna = self.compute_img_embedding(image_benna, image_benna.clone().to("cpu"), overwrite=True)
            loss_benna_ben = self.compute_embedding_loss(embeddings_benna, embeddings_ben)
            loss_benna_adv = self.compute_embedding_loss(embeddings_benna, embeddings_adv)

            total_loss += loss_benna_ben - loss_benna_adv


        return total_loss, (loss_ben_ben, loss_adv_adv, loss_ben_adv, loss_adv_ben)

    def compute_learnable_sanity_attack_loss(self, image: torch.tensor, exemplar_ben_emb: torch.tensor, exemplar_adv_emb: torch.tensor):
        """
        The function takes as input both the benign and adversarial images, both after being perturbed with the learnable noise

        """
        embeddings = self.compute_img_embedding(image, image.clone().to("cpu"), overwrite=True)

        loss_ben = self.compute_embedding_loss(embeddings, exemplar_ben_emb)
        loss_adv = self.compute_embedding_loss(embeddings, exemplar_adv_emb)


        total_loss = loss_ben - loss_adv

        return total_loss, (loss_ben, loss_adv)

    def set_default_text(self, target_label_str: str):
        self.default_text = target_label_str

    """
    NOTE: The naming of the following two functions might be misleading/ confusing
    The logits are not actually logits but embeddings, but we call them logits in order to unify the naming
    """
    def forward_logits(self, image):
        image = image.squeeze()
        return self.compute_img_embedding(image, image.clone(), overwrite=True)

    def compute_clf_loss(self, logits, target: int):
        txt_embedding = self.compute_txt_embedding(self.default_text)
        img_embedding = logits
        loss = self.compute_embedding_loss(img_embedding, txt_embedding)
        return loss
