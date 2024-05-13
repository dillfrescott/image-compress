# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from pathlib import Path
import pickle
import numpy as np

import torch
from PIL import Image
from torch import Tensor
from torchmetrics.image import (
    FrechetInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from torchvision.transforms import ToTensor
from tqdm import tqdm

from neuralcompression.metrics import (
    MultiscaleStructuralSimilarity,
    calc_psnr,
    pickle_size_of,
    update_patch_fid,
)


def rescale_image(image: Tensor, back_to_float: bool = True) -> Tensor:
    dtype = image.dtype
    image = (image * 255 + 0.5).to(torch.uint8)

    if back_to_float:
        image = image.to(dtype)

    return image


def main():
    parser = ArgumentParser()

    parser.add_argument("input_path", type=str, help="path to input PNG file or compressed file")
    parser.add_argument("output_path", type=str, help="path to output directory")
    parser.add_argument("--decompress", action="store_true", help="decompress the compressed file")

    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    device = torch.device("cuda")
    model = torch.hub.load("facebookresearch/NeuralCompression", "msillm_quality_1")
    model = model.to(device)
    model = model.eval()
    model.update()
    model.update_tensor_devices("compress")

    totensor = ToTensor()
    msssim_metric = MultiscaleStructuralSimilarity(data_range=255.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    if args.decompress:
        # Load the compressed file
        with open(input_path, "rb") as f:
            compressed = pickle.load(f)

        # Decompress the file
        with torch.no_grad():
            decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)

        # Save the decompressed image
        decompressed_pil = Image.fromarray((decompressed.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        decompressed_pil.save(output_path / f"{input_path.stem}_decompressed.png")

        print(f"Decompressed image saved to: {output_path / f'{input_path.stem}_decompressed.png'}")

    else:
        with open(input_path, "rb") as f:
            image_pil = Image.open(f)
            image_pil = image_pil.convert("RGB")

        image = totensor(image_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            compressed = model.compress(image, force_cpu=False)
            decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)

        num_bytes = pickle_size_of(compressed)
        bpp = num_bytes * 8 / (image.shape[0] * image.shape[-2] * image.shape[-1])

        orig_image = rescale_image(image.cpu())
        pred_image = rescale_image(decompressed.cpu())

        with torch.no_grad():
            update_patch_fid(image, decompressed, fid_metric)

            orig_image = rescale_image(image)
            pred_image = rescale_image(decompressed)

            psnr_val = calc_psnr(pred_image, orig_image)
            msssim_metric(pred_image, orig_image)
            lpips_metric(decompressed, image)

        # Save the compressed file
        output_file = output_path / f"{input_path.stem}.compressed"
        with open(output_file, "wb") as f:
            pickle.dump(compressed, f)

        print("Compression complete")
        print(f"Rate: {bpp}")
        print(f"PSNR: {psnr_val}")
        print(f"MS-SSIM: {msssim_metric.compute()}")
        print(f"LPIPS: {lpips_metric.compute()}")
        print(f"FID: {fid_metric.compute()}")
        print(f"Compressed file saved to: {output_file}")

if __name__ == "__main__":
    main()