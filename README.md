# YOLO Cakes: Dataset Preparation & Augmentation Pipeline

This repository provides a pipeline to prepare template images, generate masks, augment data, and set up a YOLO-compatible dataset structure for object detection tasks.

---

## Workflow Overview

1. **Prepare Template Images & Masks**
   - Run the interactive tool to add your template images and assign class names.
   - The tool uses [transparent-background](https://github.com/OPHoperHPO/transparent-background) to automatically generate object masks.

2. **Edit Training Configuration**
   - Modify `config/config.yaml` to adjust parameters such as the number of generated images, object classes, and augmentation settings.

3. **Generate Augmented Dataset**
   - Run the main setup script to create a synthetic dataset with random backgrounds, object placements, and bounding box annotations.

---

## Step-by-Step Usage

### 1. Prepare Template Images

```sh
python prepare.py
```

- Enter your **project name** (e.g., `cakes_project`).
- Add images one by one:
  - **Image path**: Full path to your image file.
  - **Class name**: Object class (e.g., `cake`, `bottle`).
- Press **Enter** on an empty line to finish and start processing.

This will:
- Copy your images to `data/{project_name}/template_img/`
- Generate masks in `data/{project_name}/template_mask/`
- Optionally copy a background image folder for augmentation.

### 2. Configure Training Settings

Edit `config/config.yaml`:

- Set `num_new_images` to the number of synthetic images to generate.
- Adjust `labels` and augmentation parameters as needed.

### 3. Generate Dataset

Run the setup script:

```sh
bash main_setup.sh
```

This will:
- Augment images by placing templates on random backgrounds.
- Generate YOLO-format bounding box annotations.
- Split the dataset into `train` and `val` folders.
- Write YOLO config files (`yolo.cfg`, `yolo.data`, `classes.txt`).

---

## Output Structure

After running the pipeline, your dataset will be organized as follows:

```
data/
  {project_name}_generated/
    train/
      images/
      labels/
    val/
      images/
      labels/
    classes.txt
    yolo.cfg
    yolo.data
    train.txt
    val.txt
```

---

## Main Scripts

- `prepare.py`: Interactive tool for template image and mask preparation.
- `main_setup.py`: Main pipeline for augmentation and dataset setup.
- `main_setup.sh`: Bash script to run the full pipeline.

---

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

---

## References

- [transparent-background](https://github.com/OPHoperHPO/transparent-background)
- [YOLOv3](https://pjreddie.com/darknet/yolo/)

---

**Quick Start Summary:**  
Run `prepare.py` → Add images & classes → Edit `config/config.yaml` → Run `main_setup.sh`  
Your YOLO dataset will be ready for training!
