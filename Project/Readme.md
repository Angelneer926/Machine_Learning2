# **WSI Patch Generation and DDPM Training**

This repository provides a pipeline for preprocessing whole slide images (WSIs), extracting patches, and training a **Denoising Diffusion Probabilistic Model (DDPM)** to generate synthetic histopathological image patches.

## **1. Virtual Environment Setup**

Before running the scripts, set up the required Conda environment:

```bash
conda env create -f env.yml
conda activate Project
```
## **2. WSI Data Preprocessing**

### **2.1 Patch Extraction**
To extract **64×64** patches from WSIs, run:

```bash
python create_patches_fp.py --source dataset_thumbnails --save_dir RESULTS_DIRECTORY --patch_size 64 --step_size 64 --preset bwh_biopsy.csv --seg --patch --stitch
```
### **Why Choose 64×64 Patches?**
- Cutting WSIs into **64×64 patches** ensures that each sample contains **30,000 to 160,000 patches**, providing sufficient training data for DDPM.
- **Lower-resolution patches** reduce training time compared to higher resolutions.
- However, **histopathology images** have intricate textures, and 64×64 patches may struggle to capture fine details. **Future iterations** may explore **128×128 resolution** for better structure preservation.

### **2.2 Organizing Patch Data**
After extraction, patches are stored within subdirectories corresponding to each WSI:

```bash
python save_slide.py
```

### **2.3 Merging Patches by Subtype**
To combine patches from different WSIs into subtypes for model training:

```bash
python patch_into_class.py
```
## **3. DDPM Model Training**
Train the diffusion model for image generation:

```bash
time python defusion_train.py
```

- Due to time constraints, the current training step count is set to **10,000** (`train_num_steps=10000`).
- In future experiments, this may be increased to **500,000** to improve generation quality.

---

## 4. Model and Output Storage
- The **`results/`** directory contains trained **DDPM models** for each subtype.
- The **`generated_images/`** directory stores the **generated images** for each subtype.

## 5. Inception Scores for Real and Generated Images
- The results are stored in the **`evaluation_results_IS/`** directory.
- The evaluate_IS.py computes inception scores(IS) for both real and generated images for each category, given that the base directories are set correctly in the code. (We could not upload datasets on Github because the size was too big.)

```
python evaluate_IS.py
```

```bash
results/            # Directory containing trained DDPM models
generated_images/   # Directory containing generated images for each subtype
evaluation_results_IS/ # Directory containing the Inception Score results for real and generated images for all categories
```