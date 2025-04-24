#!/usr/bin/env python3
import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image

# Ensure local face_alignment module is importable
base_dir = os.path.dirname(os.path.abspath(__file__))
fa_dir   = os.path.join(base_dir, 'face_alignment')
sys.path.insert(0, fa_dir)

# align.py defines get_aligned_face()
from align import get_aligned_face

def align_lfw(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Iterate identities
    for person in sorted(os.listdir(input_dir)):
        person_dir = os.path.join(input_dir, person)
        if not os.path.isdir(person_dir):
            continue
        out_person = os.path.join(output_dir, person)
        os.makedirs(out_person, exist_ok=True)
        for img_name in sorted(os.listdir(person_dir)):
            img_path = os.path.join(person_dir, img_name)
            face = get_aligned_face(img_path)
            if face is None:
                print(f"[WARN] Failed to align: {img_path}")
                continue
            out_name = os.path.splitext(img_name)[0] + '.png'
            out_path = os.path.join(out_person, out_name)
            face.save(out_path)
    print('Alignment complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Align LFW with EdgeFace PyTorch MTCNN (112×112)')
    parser.add_argument('-i', '--input',  required=True, help='Raw LFW dir (e.g. data/lfw)')
    parser.add_argument('-o', '--output', required=True, help='Aligned output dir (e.g. data/lfw_arc_112)')
    args = parser.parse_args()
    align_lfw(args.input, args.output)
