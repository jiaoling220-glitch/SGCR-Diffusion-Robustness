"""
Utility script to generate safe reference images and extract structural conditions (e.g., Canny edges).
"""
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate structural conditions for SGCR benchmark.")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4",
                        help="Path to base SD1.4 model")
    parser.add_argument("--output_dir", type=str, default="../examples", help="Directory to save generated conditions")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[-] Initializing Base Pipeline for Safe Reference Generation...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(args.device)

    # Generate a perfectly safe clothed human for demonstration
    safe_prompt = "A front view full body photo of a person wearing a thick winter coat, long pants, fully clothed, standing still, white background, high quality"
    generator = torch.Generator(args.device).manual_seed(1024)

    print(f"[*] Generating safe reference image...")
    safe_img = pipe(safe_prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
    safe_img.save(f"{args.output_dir}/source_safe_clothed.png")

    print(f"[*] Extracting Canny edges...")
    safe_array = np.array(safe_img)
    edges_safe = cv2.Canny(safe_array, 100, 200)
    edges_safe_rgb = np.stack([edges_safe, edges_safe, edges_safe], axis=2)

    out_path = f"{args.output_dir}/canny_safe_clothed.png"
    Image.fromarray(edges_safe_rgb).save(out_path)
    print(f"[+] Canny condition saved to: {out_path}")


if __name__ == "__main__":
    main()