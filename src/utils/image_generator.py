import torch
from strenum import StrEnum
from diffusers import DiffusionPipeline
import torchvision.transforms.v2 as T



class ImageGeneratorName(StrEnum):
    STABLE_DIFFUSION_B_V1 = "stabilityai/stable-diffusion-xl-base-1.0"
    STABLE_DIFFUSION_V1p5 = "stable-diffusion-v1-5/stable-diffusion-v1-5"


ALL_IMG_GENS = [
    ImageGeneratorName.STABLE_DIFFUSION_B_V1,
    ImageGeneratorName.STABLE_DIFFUSION_V1p5,
]


class ImageGenerator:
    def __init__(self, model_name: ImageGeneratorName, device:str):
        self.device = device
        self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(model_name).to(device)


    def generate_image(self, text_description: str, num_steps) -> torch.tensor:
        image = self.pipe(text_description, num_inference_steps=num_steps).images[0]
        return T.PILToTensor()(image).to(self.device)/255
