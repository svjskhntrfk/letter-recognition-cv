# Noisy Letter Grid OCR — Counting 10,000 Letters from a Single Image

This repository demonstrates an end-to-end computer vision pipeline for counting occurrences of A–Z letters in a single large noisy image.

The input image contains a **100×100 grid (10,000 tiles)**. Each tile includes one uppercase letter with **random rotation, translation, scale, and color**, plus **clutter from semi-transparent circles**. The goal is to produce a `letters.csv` file with per-class counts.

---

## Problem

Given a large image with many letters and strong synthetic noise:
- letters are **uppercase A–Z**
- **font size varies** 
- letters can be **rotated** and **shifted within each tile**
- each tile includes **10–40 semi-transparent circles** 
- there are **10,000 letters** total

**Input:** You can download the example image here: [Google Drive — letters image](https://drive.google.com/file/d/15XPScSlkhXaBvoNBKKT0zxVeTatJr5sO/view?usp=sharing)

**Output:** counts of each letter in a simple CSV.

---

## Key idea

Instead of treating this as generic OCR, I exploit the structure of the data:

### 1) Deterministic tiling, so no detector required
The image is a regular grid. I infer that it consists of:
- **100 tiles horizontally**
- **100 tiles vertically**
- all tiles have the same size

This allows fast, deterministic cropping of all 10,000 candidate letter tiles.

### 2) Synthetic training data + target-like validation
There is no labeled dataset for the target image, so I generate a labeled training set synthetically from base templates (A–Z), then apply augmentations to match the target corruption.
I also generate a **fixed, reproducible validation set** designed to be as close as possible to the real image distribution.

### 3) Label-conditional augmentations
A central part of the pipeline is a **label-adaptive augmentation strategy**:
- “fragile” letters (thin strokes or sharp corners) receive milder occlusion
- “round” letters tolerate stronger perturbations
- circle clutter is added with class-dependent probability
- one-of degradation sampling (none / hide a part of the letter / blur) prevents overfitting to a single corruption type

This was particularly useful for reducing shortcut learning and class-bias issues such as the tendency to over-predict **I** for letters with straight strokes (e.g., **L/J**).

### 4) CNN classifier + full-grid inference
A CNN classifier predicts A–Z for each tile. Predictions are aggregated into counts and exported as `letters.csv`.
I also run a quick qualitative check by plotting random tiles with predicted labels to catch pipeline bugs early.

---

## How to run (high-level)

This project was developed in a notebook environment (Kaggle/Colab-style).  
To reproduce:

1. Open the notebook and set the path to the input image.
2. Run cells in the given order in notebook

Dependencies:
- Python 3.10+
- PyTorch + torchvision
- PyTorch Lightning
- PIL, numpy, matplotlib

---

## Notes on tool assistance

I used ChatGPT as an assistant for code cleanup and refactoring (improving readability, reorganizing functions, naming, and documentation).  
All modeling choices, experimental iterations, and final outputs were designed and verified by me.

---

## What this project demonstrates

- building a  CV pipeline end-to-end (
- data-centric ML (augmentations, target-distribution validation, failure-mode analysis)
- practical robustness techniques (occlusion, clutter simulation, class-dependent augmentation)
- reproducible dataset generation and validation
