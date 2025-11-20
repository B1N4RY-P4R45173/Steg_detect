#!/usr/bin/env python3
"""
Data preprocessing utilities for steganography detection
"""

import os
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

def resize_images(input_dir, output_dir, size=(256, 256)):
    """
    Resize all images in a directory

    Args:
        input_dir: Input directory
        output_dir: Output directory
        size: Target size (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)

    extensions = ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']
    image_files = []

    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))

    print(f"Resizing {len(image_files)} images to {size}...")

    for i, img_path in enumerate(sorted(image_files)):
        try:
            img = Image.open(img_path)
            img_resized = img.resize(size, Image.LANCZOS)

            output_path = os.path.join(output_dir, img_path.name)
            img_resized.save(output_path)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"✓ Resized images saved to {output_dir}/")

def convert_to_grayscale(input_dir, output_dir):
    """
    Convert all images to grayscale

    Args:
        input_dir: Input directory
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []

    for ext in extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))

    print(f"Converting {len(image_files)} images to grayscale...")

    for i, img_path in enumerate(sorted(image_files)):
        try:
            img = Image.open(img_path).convert('L')
            output_path = os.path.join(output_dir, img_path.stem + '.png')
            img.save(output_path)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"✓ Grayscale images saved to {output_dir}/")

def analyze_dataset(directory):
    """
    Analyze dataset statistics

    Args:
        directory: Directory to analyze
    """
    extensions = ['.pgm', '.png', '.jpg', '.jpeg', '.bmp']
    image_files = []

    for ext in extensions:
        image_files.extend(Path(directory).glob(f'*{ext}'))

    if not image_files:
        print(f"❌ No images found in {directory}")
        return

    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {directory}")
    print('='*60)
    print(f"Total images: {len(image_files)}")

    # Sample first image for dimensions
    sample_img = Image.open(image_files[0])
    print(f"Sample dimensions: {sample_img.size}")
    print(f"Sample mode: {sample_img.mode}")

    # File size statistics
    sizes = [os.path.getsize(f) for f in image_files]
    print(f"\nFile size statistics:")
    print(f"  Min: {min(sizes) / 1024:.2f} KB")
    print(f"  Max: {max(sizes) / 1024:.2f} KB")
    print(f"  Mean: {np.mean(sizes) / 1024:.2f} KB")
    print(f"  Total: {sum(sizes) / (1024**2):.2f} MB")

    # Image dimensions
    dimensions = set()
    modes = set()

    for img_path in image_files[:100]:  # Sample first 100
        try:
            img = Image.open(img_path)
            dimensions.add(img.size)
            modes.add(img.mode)
        except:
            pass

    print(f"\nUnique dimensions (sample): {dimensions}")
    print(f"Image modes (sample): {modes}")
    print('='*60 + '\n')

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for steganalysis')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Resize command
    resize_parser = subparsers.add_parser('resize', help='Resize images')
    resize_parser.add_argument('--input', '-i', required=True, help='Input directory')
    resize_parser.add_argument('--output', '-o', required=True, help='Output directory')
    resize_parser.add_argument('--width', type=int, default=256, help='Target width')
    resize_parser.add_argument('--height', type=int, default=256, help='Target height')

    # Grayscale command
    gray_parser = subparsers.add_parser('grayscale', help='Convert to grayscale')
    gray_parser.add_argument('--input', '-i', required=True, help='Input directory')
    gray_parser.add_argument('--output', '-o', required=True, help='Output directory')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument('--directory', '-d', required=True, help='Directory to analyze')

    args = parser.parse_args()

    if args.command == 'resize':
        resize_images(args.input, args.output, (args.width, args.height))

    elif args.command == 'grayscale':
        convert_to_grayscale(args.input, args.output)

    elif args.command == 'analyze':
        analyze_dataset(args.directory)

    else:
        parser.print_help()

if __name__ == '__main__':
    main()
