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


## 4. Inception Scores for Real and Generated Images
- The results are stored in the **`evaluation_results_IS/`** directory.
- The Inceptions Scores(IS) are the evaluation metric for the quality of the generated images because higher IS indicates that the classification are more confident and the image dataset is diverse. 
- We compare the IS of the generated images and real images. We aim to generate images such that the IS of the generated and real images are similar.
- For the **Code & Graphs Milestone I**, we computed IS for all real images and 1,000 generated images across all categories.
- For the **Code & Graphs Milestone II**, We computed IS for all real images, 1,000 generated images  across all categories, and 200 generated images with finetuning for the minority classes: LGSC, CC, and MC.
- With finetuning, Inception Scores has been improved for the minority classes.
- The evaluate_IS.py computes inception scores(IS) for both real and generated images for each category, given that the base directories are set correctly in the code. (We could not upload datasets on Github because the size was too big.)

```
python evaluate_IS.py
```
## 5. Combine Patches
```
patch python combine.py
```
## 6. Feature Extraction

(1) generated data:

```
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir combined_h5 --data_slide_dir combined_images --csv_path image_labels.csv --feat_dir FEATURES_DIRECTORY1 --batch_size 4096 --slide_ext .png
```

(2) real data:

```
python get_step_featcsv.py
CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir RESULTS_DIRECTORY --data_slide_dir dataset_thumbnails --csv_path RESULTS_DIRECTORY/Step_2.csv --feat_dir FEATURES_DIRECTORY2 --batch_size 4096 --slide_ext .png
```

## 7. Training Splits

```
python get_labelcsv.py
python create_splits_seq.py --task task_1_tumor_subtyping_5_classes --seed 1 --k 10
```

## 8. Training for Subtyping Problems

```
CUDA_VISIBLE_DEVICES=0 python main.py --drop_out 0.25 --early_stopping --lr 2e-4 --k 10 --exp_code task_1_tumor_subtyping_5_classes --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_subtyping_5_classes --model_type clam_sb --log_data --subtyping --data_root_dir ./ --embed_dim 1024
```

---
## 9. Model and Output Storage
- The **`results/`** directory contains trained **DDPM models** for each subtype.
- The **`generated_images/`** directory stores the **generated images** for each subtype.

```bash
results/            # Directory containing trained DDPM models
generated_images/   # Directory containing generated images for each subtype
evaluation_results_IS/ # Directory containing the Inception Score results for real and generated images for all categories
```
