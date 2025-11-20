#!/usr/bin/env python3
"""
Script to create steganographic images using LSB embedding
For more advanced algorithms (HUGO, S-UNIWARD), consider using specialized tools
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import random

def lsb_embed(image_array, message, num_lsb=1):
    """
    Embed message using Least Significant Bit (LSB) steganography

    Args:
        image_array: numpy array of image
        message: binary message to embed
        num_lsb: number of LSB to use (1-3)

    Returns:
        Modified image array with embedded message
    """
    # Flatten image
    flat_image = image_array.flatten()

    # Convert message to binary if it's a string
    if isinstance(message, str):
        message = ''.join(format(ord(c), '08b') for c in message)

    # Embed message length first
    msg_len = len(message)
    len_binary = format(msg_len, '032b')

    # Combine length + message
    full_binary = len_binary + message

    # Embed in LSB
    for i in range(len(full_binary)):
        if i >= len(flat_image):
            break

        # Clear LSB
        flat_image[i] = (flat_image[i] & ~1) | int(full_binary[i])

    return flat_image.reshape(image_array.shape)

def random_lsb_embed(image_array, payload_ratio=0.4):
    """
    Embed random bits to simulate steganography

    Args:
        image_array: numpy array of image
        payload_ratio: fraction of pixels to modify (0.0 to 1.0)

    Returns:
        Modified image array
    """
    # Create a copy
    stego = image_array.copy()

    # Flatten
    flat = stego.flatten()

    # Number of pixels to modify
    num_modify = int(len(flat) * payload_ratio)

    # Random indices
    indices = random.sample(range(len(flat)), num_modify)

    # Flip LSB randomly
    for idx in indices:
        flat[idx] = flat[idx] ^ 1  # XOR with 1 to flip LSB

    return flat.reshape(image_array.shape)

def create_stego_images(input_dir, output_dir, algorithm='LSB', num_images=None):
    """
    Create steganographic images from cover images

    Args:
        input_dir: Directory containing cover images
        output_dir: Directory to save stego images
        algorithm: Embedding algorithm ('LSB' or 'RANDOM')
        num_images: Number of images to process (None = all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of images
    extensions = ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))

    image_files = sorted(image_files)

    if num_images:
        image_files = image_files[:num_images]

    print(f"Processing {len(image_files)} images with {algorithm} algorithm...")

    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path).convert('L')
            img_array = np.array(img, dtype=np.uint8)

            # Embed message
            if algorithm.upper() == 'LSB':
                # Create random message
                message_len = random.randint(100, 1000)
                message = ''.join(random.choice('01') for _ in range(message_len))
                stego_array = lsb_embed(img_array, message)

            elif algorithm.upper() == 'RANDOM':
                # Random LSB flipping
                payload = random.uniform(0.3, 0.5)
                stego_array = random_lsb_embed(img_array, payload)

            else:
                print(f"❌ Unknown algorithm: {algorithm}")
                return

            # Save stego image
            stego_img = Image.fromarray(stego_array.astype(np.uint8))
            output_path = os.path.join(output_dir, img_path.name)
            stego_img.save(output_path)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue

    print(f"\n✓ Created {len(image_files)} stego images in {output_dir}/")

def main():
    parser = argparse.ArgumentParser(
        description='Create steganographic images from cover images'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory with cover images')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for stego images')
    parser.add_argument('--algorithm', '-a', default='RANDOM',
                        choices=['LSB', 'RANDOM'],
                        help='Embedding algorithm (default: RANDOM)')
    parser.add_argument('--num', '-n', type=int, default=None,
                        help='Number of images to process (default: all)')

    args = parser.parse_args()

    # Verify input directory exists
    if not os.path.isdir(args.input):
        print(f"❌ Input directory not found: {args.input}")
        return

    # Create stego images
    create_stego_images(
        args.input,
        args.output,
        args.algorithm,
        args.num
    )

    print("\n" + "="*60)
    print("Note: For advanced algorithms (HUGO, S-UNIWARD, WOW),")
    print("consider using specialized tools like:")
    print("  - DDE Lab tools: https://dde.binghamton.edu/download/")
    print("  - Aletheia: https://github.com/daniellerch/aletheia")
    print("="*60)

if __name__ == '__main__':
    main()
