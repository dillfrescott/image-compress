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

    parser.add_argument("input_dir", type=str, help="path to input directory containing PNG files")
    parser.add_argument("output_dir", type=str, help="path to output directory")
    parser.add_argument("--decompress", action="store_true", help="decompress the compressed files")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

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
        for file in tqdm(input_dir.glob("*.compressed")):
            with open(file, "rb") as f:
                compressed = pickle.load(f)

            with torch.no_grad():
                decompressed = model.decompress(compressed, force_cpu=False).clamp(0.0, 1.0)

            decompressed_pil = Image.fromarray((decompressed.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            decompressed_pil.save(output_dir / f"{file.stem}_decompressed.png")

            print(f"Decompressed image saved to: {output_dir / f'{file.stem}_decompressed.png'}")

    else:
        for file in tqdm(input_dir.glob("*.png")):
            with Image.open(file) as image_pil:
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
            output_file = output_dir / f"{file.stem}.compressed"
            with open(output_file, "wb") as f:
                pickle.dump(compressed, f)

            print(f"Compression complete for {file.stem}")
            print(f"Rate: {bpp}")
            print(f"PSNR: {psnr_val}")
            print(f"MS-SSIM: {msssim_metric.compute()}")
            print(f"LPIPS: {lpips_metric.compute()}")
            print(f"FID: {fid_metric.compute()}")
            print(f"Compressed file saved to: {output_file}")

if __name__ == "__main__":
    main()