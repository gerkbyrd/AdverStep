import torch
from .vlm import VLM
from .defended_vlm import DefendedVLM
from .classifier import ImageClassifier, DefendedImageClassifier

def evaluate_image(
        image: torch.tensor,
        clf: ImageClassifier,
        vlm: VLM,
        user_query: str,
        prompt_image: torch.tensor = None,
        additive_image: torch.tensor = None,
        printer: bool=True,
        vlmout: bool=False,
):
    image_q = image.clone().type(torch.uint8)

    #print("Evaluating float image ...")
    if not vlmout:
        logs=eval_image(image, clf, vlm, user_query, prompt_image, additive_image, printer, vlmout)
        logsq, responseq = eval_image(image_q, clf, vlm, user_query, prompt_image, additive_image, printer, vlmout)
    else:
        logs, response=eval_image(image, clf, vlm, user_query, prompt_image, additive_image, printer, vlmout)
        logsq, responseq = eval_image(image_q, clf, vlm, user_query, prompt_image, additive_image, printer, vlmout)

    #print("Evaluating quantized image ...")

    if not vlmout:
        return logs
    else:
        return logs, [response.split('Assistant: ')[-1], responseq.split('Assistant: ')[-1]]


def eval_image(
        image: torch.tensor,
        clf: ImageClassifier,
        vlm: VLM,
        user_query: str,
        prompt_image: torch.tensor = None,
        additive_image: torch.tensor = None,
        printer: bool =True,
        vlmout: bool=False,
        simcheck: bool =False,
):
    prompt = vlm.get_test_prompt(user_query)

    if isinstance(clf, DefendedImageClassifier):
        sampled_image=clf.preprocess(image.clone().type(torch.uint8))
        logits, label_idx, label_str = clf.classify(sampled_image, additive_image=additive_image)
    else:
        logits, label_idx, label_str = clf.classify(image)
    if isinstance(vlm, DefendedVLM):
        vlm_answer = vlm.generate(image, formatted_prompt=prompt, prompt_image=prompt_image, overwrite=True)[0]
    else:
        if "Qwen" in vlm.name or "llava" in vlm.name:
            pass
            #image=image/255
        vlm_answer = vlm.generate(image, formatted_prompt=prompt, overwrite=True)[0]
    if printer:
        print(f"Classifier -> idx: {label_idx}, str: {label_str}")
        print(f"VLM -> {vlm_answer}")
    if not vlmout:
        return logits.detach()
    else:
        return logits.detach(), vlm_answer
