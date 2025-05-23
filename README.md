OLO Synthetic Dataset Preparation and Training Pipeline
This repository provides a pipeline to prepare template images, generate masks, augment data, and set up a YOLO training dataset for custom object detection tasks.

Features
Interactive Template Preparation: Easily add images and assign class names, with automatic background removal using transparent-background.
Automatic Mask Generation: Generates binary masks for each template image.
Configurable Data Augmentation: Randomly places templates on backgrounds with configurable transformations.
YOLO Dataset Structure: Automatically splits data into train/val sets and generates YOLO config files.
Verification Tools: Visualize masks and overlays for quality control.
Quick Start
1. Prepare Template Images
Run the interactive preparation tool:
python prepare.py

Enter your project name (e.g., cakes).
Add images one by one:
Enter the full path to each image.
Assign a class name (e.g., cake, bottle).
Press Enter on an empty prompt to finish.
This will:

Copy your images to data/{project_name}/template_img/
Generate masks in data/{project_name}/template_mask/
Copy background images to data/{project_name}/background/ (auto-detected or specify with --background_path)
Optionally update config.yaml with your project and class names.
2. Configure Training Settings
Edit config.yaml:

Set num_new_images to the number of synthetic images you want to generate.
Adjust augmentation parameters as needed.
3. Generate Dataset
Run the main setup script:
bash main_setup.sh

This will:

Augment images and generate YOLO-format dataset in data/{project_name}_generated/
Split data into train and val folders
Generate classes.txt, train.txt, val.txt, and YOLO config files
File Structure
Advanced Usage
Batch Mode: You can run prepare.py with arguments for batch processing:
Verification: Use the --create_verification flag to generate overlay images for mask quality checking.
Requirements
See requirements.txt for all dependencies. Install with:

References
transparent-background
YOLOv3
Workflow Summary
Run prepare.py → Add images & classes → Masks and folders auto-generated.
Edit config.yaml → Set project name, labels, and augmentation settings.
Run main_setup.sh → Generates YOLO dataset and config files.
For more details, see the comments in flow.txt and the docstrings in each script.

Happy training!

