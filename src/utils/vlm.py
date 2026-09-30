from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForImageTextToText
import torch
import torchvision.transforms as T
from .image_utils import process_image
from strenum import StrEnum

# candidate models
class VLMName(StrEnum):
    SMOLVLM_1_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"
    SMOLVLM_1_500M = "HuggingFaceTB/SmolVLM-500M-Instruct"
    SMOLVLM_1_2B = "HuggingFaceTB/SmolVLM-Instruct"
    # SMOLVLM_2_2B = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    #SMOLVLM_2_1p7B = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    QWEN_2p5_VL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
    QWEN_2p5_VL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
    LLAVA_ONEVISION_0p5B = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    INTERNVL_3_1B = "OpenGVLab/InternVL3-1B-hf"
    INTERNVL_3_2B = "OpenGVLab/InternVL3-2B-hf"
    INTERNVL_3_8B = "OpenGVLab/InternVL3-8B-hf"

SMOL_VLMS = [
    VLMName.SMOLVLM_1_256M,
    VLMName.SMOLVLM_1_500M,
    VLMName.SMOLVLM_1_2B,
    # VLMName.SMOLVLM_2_2B,
]

QWEN_VLMS = [
    VLMName.QWEN_2p5_VL_3B,
    VLMName.QWEN_2p5_VL_7B,
]

ONEVISION_VLMS = [
    VLMName.LLAVA_ONEVISION_0p5B
]

INTERN_VLMS = [
    VLMName.INTERNVL_3_1B,
    VLMName.INTERNVL_3_2B,
    VLMName.INTERNVL_3_8B,
]

VLMS_WITH_FAST_PROCESSOR = [
    VLMName.QWEN_2p5_VL_3B,
    VLMName.QWEN_2p5_VL_7B,
    VLMName.LLAVA_ONEVISION_0p5B,
]
VLMS_WITH_FAST_PROCESSOR.extend(INTERN_VLMS)

ALL_VLMS = [
    VLMName.QWEN_2p5_VL_3B,
    VLMName.QWEN_2p5_VL_7B,
    VLMName.LLAVA_ONEVISION_0p5B,
    VLMName.SMOLVLM_1_256M,
    VLMName.SMOLVLM_1_500M,
    VLMName.SMOLVLM_1_2B,
]

VLMS = [
    VLMName.SMOLVLM_1_256M,#OG (300M)
    VLMName.SMOLVLM_1_500M,#500M
    VLMName.SMOLVLM_1_2B,#2B
    VLMName.QWEN_2p5_VL_3B,#4B
    VLMName.LLAVA_ONEVISION_0p5B,#0.9B
    VLMName.INTERNVL_3_2B,
    #VLMName.INTERNVL_2p5_2B,
]
class VLM():
    def __init__(self, model_name, device, quantize=False, attn_implementation: str = None):
        self.name = model_name
        self.device = device

        quantization_config = BitsAndBytesConfig(load_in_4bit=True) if quantize else None

        if self.name not in INTERN_VLMS:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else "auto",
                quantization_config=quantization_config,
                _attn_implementation="eager").to(device)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "mps" else "auto",
                quantization_config=quantization_config,
                _attn_implementation="eager").to(device)
        # potentially use:
        #_attn_implementation="flash_attention_2" if device == "cuda" else "eager",
        #if attn_implementation:
            #self.model.set_attn_implementation(attn_implementation)

        self.tokenizer = None

        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        try:
            self.processor.image_processor.do_image_splitting = False
            if self.processor.image_processor.resample == 1:
                self.processor.image_processor.resample = 3 # change from LANCZOS (1) to BICUBIC (3) since the former has no pytorch implementation
        except:
            print("weird img processor...")

        if model_name in INTERN_VLMS:
            self.processor.image_processor.min_patches, self.processor.image_processor.max_patches = 0, 0

        self.model.eval()


    def get_classification_loss(self, image: torch.tensor, config):
        prompt, target_tokens = self.get_training_prompt(config.user_query, config.classification_target.target_class_str)
        vlm_output = self.forward(image, prompt, overwrite=True)
        loss = self.compute_gen_loss(vlm_output, target_tokens)
        return loss

    def get_qa_loss(self, image: torch.tensor, query: str, answer: str):
        prompt, target_tokens = self.get_training_prompt(query, answer)
        vlm_output = self.forward(image, prompt, overwrite=True)
        loss = self.compute_gen_loss(vlm_output, target_tokens)
        return loss

    def get_user_message(self, user_query):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image"},
            ]
        }

    def get_test_prompt(self, user_query, add_gen=True):
        messages = [
            self.get_user_message(user_query)
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=add_gen)
        return prompt

    def get_target_tokens(self, target_generation: str):

        msg_without_template = " " + target_generation
        raw_target_tokens = self.processor(text=msg_without_template, return_tensors="pt").to(self.device)['input_ids'][0]

        msg_with_template = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_generation}
                    ]
                },
            ]

        prompt_with_template = self.processor.apply_chat_template(msg_with_template, add_generation_prompt=False)
        full_tokens = self.processor(text=prompt_with_template, return_tensors="pt").to(self.device)['input_ids'][0]

        target_tokens = full_tokens[-len(raw_target_tokens)-2:]

        return target_tokens



    def get_training_prompt(self,
            user_query: str,
            target_generation: str,
        ):
        """
        builds the prompt skeleton for the VLM including the image placeholder, the user query, and the required response
        """
        messages = [
            self.get_user_message(user_query),
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_generation}
                ]
            },
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        target_tokens = self.get_target_tokens(target_generation)
        return prompt, target_tokens


    def create_vlm_inputs(self, image: torch.tensor, formatted_prompt, overwrite: bool = False):
        """
        NOTE: padding_side should be set to "left", otherwise this will interfere with the attack optimization
        """
        if isinstance(formatted_prompt, str): formatted_prompt = [formatted_prompt]

        mock_image = T.ToPILImage()(image)
        if self.name in VLMS_WITH_FAST_PROCESSOR:
            mock_image = image

        # process
        inputs = self.processor(text=formatted_prompt, images=mock_image, return_tensors="pt", truncation=True, padding=True, padding_side="left").to(self.device)

        # convert pixels to floats (maybe not needed)
        inputs["pixel_values"] = inputs["pixel_values"].float()

        # overwrite with exact adversarial image tensor if needed
        if overwrite and self.name not in VLMS_WITH_FAST_PROCESSOR:
            image_ppd = process_image(image, self)
            for i in range(inputs['pixel_values'].shape[0]):
                inputs['pixel_values'][i][0] = image_ppd

        return inputs

    @torch.no_grad()
    def generate(self, image: torch.tensor, formatted_prompt, overwrite: bool = False, max_new_tokens=30, do_sample=True, temperature=0.5):
        inputs = self.create_vlm_inputs(image, formatted_prompt, overwrite)
        if temperature:
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
        else:
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

    #@torch.no_grad()
    def generate_greedy(self, image: torch.tensor, user_query: str, overwrite: bool = False, max_new_tokens=30, dist=False):
        fprompt = self.get_test_prompt(user_query)
        prompt = self.create_vlm_inputs(image.clone().detach(), fprompt, overwrite=True)
        clo=0
        while True:
            clo+=1
            outlog=self.model(input_ids=prompt['input_ids'], pixel_values = prompt['pixel_values']).logits[:,-1,:]
            next_token = outlog.argmax(-1).unsqueeze(0)
            if next_token.item() == self.processor.tokenizer.eos_token_id or clo == max_new_tokens:
                if dist:
                    #last_ids = prompt['input_ids']
                    nprompt = self.create_vlm_inputs(image, fprompt, overwrite=True)
                    #prompt['input_ids'] = last_ids
                    #print(nprompt['pixel_values'])
                    #input(prompt['pixel_values'])
                    #input(torch.autograd.grad(prompt['pixel_values'].sum(), image))
                    outlog=self.model(input_ids=prompt['input_ids'], pixel_values = nprompt['pixel_values']).logits[:,-1,:]
                    dima=torch.exp(outlog).squeeze(0)
                break
            prompt['input_ids'] = torch.concatenate([prompt['input_ids'],next_token], dim=1)
        generated_texts = self.processor.batch_decode(prompt['input_ids'], skip_special_tokens=True)
        if dist:
            return [o.split(self.get_vlm_assistant_delimiter())[-1] for o in generated_texts], dima
        return [o.split(self.get_vlm_assistant_delimiter())[-1] for o in generated_texts]



    def forward(self, image, formatted_prompt, overwrite: bool = False, output_attentions: bool = False, output_hidden_states: bool = False):
        inputs = self.create_vlm_inputs(image, formatted_prompt, overwrite)
        out = self.model(**inputs, use_cache=False, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        return out

    def forward_logits(self, image, formatted_prompt = None):
        image = image.squeeze()
        if not formatted_prompt: formatted_prompt = self.default_training_prompt
        vlm_outputs = self.forward(image, formatted_prompt, overwrite=True)
        return vlm_outputs.logits

    def set_default_prompt_info(self, user_query: str, target_label_str: str):
        self.user_query = user_query
        self.default_training_prompt, self.target_tokens = self.get_training_prompt(user_query, target_label_str)
        self.default_test_prompt = self.get_test_prompt(user_query)

    def compute_clf_loss(self, logits, target_label_str: str):
        logits_to_optimize = logits[:,-len(self.target_tokens)-1:-1,:].transpose(1,2)
        target_tokens = self.target_tokens.unsqueeze(0).repeat(logits_to_optimize.shape[0], 1)
        return torch.nn.CrossEntropyLoss()(logits_to_optimize, target_tokens)


    def compute_gen_loss(self, vlm_output, target_tokens):
        logits_to_optimize = vlm_output.logits[:,-len(target_tokens)-1:-1,:].transpose(1,2)
        target_tokens = target_tokens.unsqueeze(0).repeat(logits_to_optimize.shape[0], 1)
        return torch.nn.CrossEntropyLoss()(logits_to_optimize, target_tokens)

    @torch.no_grad()
    def generate_e2e(self, image:torch.tensor, user_query: str, max_new_tokens: int = 30, temperature: float = 0.5):
        test_prompt = self.get_test_prompt(user_query)
        #new: allow T=0 in a fast way (we still have to do slow if we use the last token logits for detection)
        do_sample = temperature > 0
        if not do_sample:
            temperature = 0.5
        output = self.generate(image, test_prompt, overwrite=True, max_new_tokens=max_new_tokens, do_sample = do_sample, temperature=temperature)

        answers = [o.split(self.get_vlm_assistant_delimiter())[-1] for o in output]
        return answers

    @torch.no_grad()
    def forward_e2e(self, image:torch.tensor, user_query: str, overwrite: bool = False, output_attentions: bool = False, output_hidden_states: bool = False):
        test_prompt = self.get_test_prompt(user_query)
        inputs = self.create_vlm_inputs(image, test_prompt, overwrite)
        out = self.model(**inputs, use_cache=False, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        return out

    def get_vlm_assistant_delimiter(self):
        if self.name in SMOL_VLMS:
            return "Assistant: "
        else:
            # valid for Qwen2.5, InternVL3
            return "assistant\n"
