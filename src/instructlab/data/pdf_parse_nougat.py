"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from functools import partial
import pypdfium2
import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible, close_envs
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import ImageDataset
from nougat.dataset.rasterize import rasterize_paper
from tqdm import tqdm

model = None

def load_model():
    global model
    if model == None:
        checkpoint = get_checkpoint(None, model_tag="0.1.0-small")
        print("Loading nougat checkpoint:", checkpoint)
        model = NougatModel.from_pretrained(checkpoint)
        model.eval()

def pdf_to_markdown(file) -> str:
    load_model()

    with open(file, 'rb') as f:
        pdfbin = f.read()
    pdf = pypdfium2.PdfDocument(pdfbin)

    images = rasterize_paper(pdf)

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
    )

    result = ""
    for idx, sample in tqdm(enumerate(dataloader), total=len(dataloader)):
        if sample is None:
            continue
        model_output = model.inference(image_tensors=sample)
        for output in model_output["predictions"]:
            page = markdown_compatible(output)
            result += page

    return result

if __name__ == "__main__":
    result = pdf_to_markdown("/home/jsightle/tmp/btf/backToTheFuture/data.pdf")
    print(result)