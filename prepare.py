#!/usr/bin/env python3
"""
Template Image and Mask Preparation Tool
Automatically processes uploaded images, renames them with class names and indices,
and generates masks using transparent-background for template preparation.
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import shutil
import yaml
import re
from transparent_background import Remover

class TemplatePreparator:
    def __init__(self, project_folder, copy_background=True, background_source=None):
        """
        Initialize the template preparator
        
        Args:
            project_folder (str): Main folder name to store all processed data
            copy_background (bool): Whether to copy background folder
            background_source (str): Path to source background folder
        """
        # Create the project folder inside data/ directory
        self.project_folder = os.path.join("data", project_folder)
        self.template_img_folder = os.path.join(self.project_folder, "template_img")
        self.template_mask_folder = os.path.join(self.project_folder, "template_mask")
        self.class_counters = {}  # Track indices for each class
        
        # Initialize transparent-background remover
        self.remover = None
        self._init_remover()
        
        # Create necessary folders and optionally copy background
        if copy_background:
            status = self.setup_project_structure(background_source)
        else:
            self.create_folders()
    
    def _init_remover(self):
        try:
            print("Initializing transparent-background model...")
            # You can customize these settings based on your needs
            self.remover = Remover(mode='base')  # options: 'base', 'fast', 'base-nightly'
            print("Model loaded successfully!")
        except ImportError:
            print("Error: transparent-background is not installed. Please install it with:")
            print("pip install transparent-background")
            raise
        except Exception as e:
            print(f"Error initializing transparent-background model: {e}")
            raise
    
    def create_folders(self):
        """Create the required folder structure"""
        os.makedirs(self.project_folder, exist_ok=True)
        os.makedirs(self.template_img_folder, exist_ok=True)
        os.makedirs(self.template_mask_folder, exist_ok=True)
        print(f"Created project structure in: {self.project_folder}")
        print(f"  - Template images: {self.template_img_folder}")
        print(f"  - Template masks: {self.template_mask_folder}")
    
    def copy_background_folder(self, source_background_path=None):
        """
        Copy background folder from source location to project folder
        
        Args:
            source_background_path (str): Path to source background folder. 
                                        If None, will try to find it automatically.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Try to find background folder automatically if not provided
        if source_background_path is None:
            # Look for background folder in common locations
            possible_paths = [
                # Look in parent directory structure
                os.path.join("data", "background"),
                os.path.join("..", "data", "background"),
                os.path.join("..", "..", "data", "background"),
                # Look in same level as current project
                os.path.join(os.path.dirname(self.project_folder), "background"),
                # Look for any existing yolo project with background
                "C:\\Users\\goaic\\Desktop\\yolo_cakes\\data\\background"  # Your specific path
            ]
            
            source_background_path = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    source_background_path = path
                    print(f"Found background folder at: {path}")
                    break
            
            if source_background_path is None:
                print("Warning: Background folder not found automatically.")
                print("Please specify the path manually using --background_path argument")
                return False
        
        # Validate source path
        if not os.path.exists(source_background_path):
            print(f"Error: Source background folder not found: {source_background_path}")
            return False
        
        if not os.path.isdir(source_background_path):
            print(f"Error: Path is not a directory: {source_background_path}")
            return False
        
        # Define destination path
        dest_background_path = os.path.join(self.project_folder, "background")
        
        try:
            # Copy the entire background folder
            if os.path.exists(dest_background_path):
                print(f"Background folder already exists at: {dest_background_path}")
                overwrite = input("Overwrite existing background folder? (y/n): ").lower().startswith('y')
                if overwrite:
                    shutil.rmtree(dest_background_path)
                else:
                    print("Skipping background folder copy.")
                    return True
            
            print(f"Copying background folder...")
            print(f"  From: {source_background_path}")
            print(f"  To: {dest_background_path}")
            
            shutil.copytree(source_background_path, dest_background_path)
            
            # Count copied files
            bg_files = []
            for root, dirs, files in os.walk(dest_background_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        bg_files.append(file)
            
            print(f"Successfully copied background folder with {len(bg_files)} images")
            return True
            
        except Exception as e:
            print(f"Error copying background folder: {e}")
            return False

    def setup_project_structure(self, source_background_path=None):
        """
        Set up project structure with only template folders and optional background
        
        Args:
            source_background_path (str): Path to source background folder
        
        Returns:
            dict: Status of folder creation
        """
        status = {
            'folders_created': [],
            'background_copied': False,
            'success': True
        }
        
        try:
            # Create only the essential folders
            folders_to_create = [
                self.project_folder,
                self.template_img_folder,
                self.template_mask_folder
            ]
            
            for folder in folders_to_create:
                os.makedirs(folder, exist_ok=True)
                status['folders_created'].append(folder)
            
            print("Created template project structure:")
            for folder in status['folders_created']:
                print(f"  - {folder}")
            
            # Copy background folder if requested
            status['background_copied'] = self.copy_background_folder(source_background_path)
            
            return status
            
        except Exception as e:
            print(f"Error setting up project structure: {e}")
            status['success'] = False
            return status
    
    def generate_filename(self, class_name, extension=".jpg"):
        """
        Generate filename with auto-incrementing index
        
        Args:
            class_name (str): Class name (e.g., 'cake', 'bottle')
            extension (str): File extension
            
        Returns:
            str: Generated filename (e.g., 'cake_1.jpg', 'cake_2.jpg')
        """
        if class_name not in self.class_counters:
            self.class_counters[class_name] = 0
        
        self.class_counters[class_name] += 1
        return f"{class_name}_{self.class_counters[class_name]}{extension}"
    
    def remove_background(self, input_image):
        """
        Remove background from image using transparent-background
        
        Args:
            input_image: PIL Image or numpy array
            
        Returns:
            tuple: (foreground_image, mask_image) both as numpy arrays
        """
        try:
            # Convert to PIL if needed
            if isinstance(input_image, np.ndarray):
                # Convert BGR to RGB for PIL
                input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_pil = Image.fromarray(input_rgb)
            else:
                input_pil = input_image.convert('RGB')
            
            # Process with transparent-background
            # This returns RGBA image with transparent background
            output_pil = self.remover.process(input_pil, type='rgba')
            
            # Convert to numpy array
            output_np = np.array(output_pil)
            
            # Extract alpha channel as mask and RGB as foreground
            if output_np.shape[2] == 4:  # RGBA
                mask = output_np[:, :, 3]  # Alpha channel (0-255)
                foreground_rgb = output_np[:, :, :3]  # RGB channels
            else:
                # Fallback: create mask based on non-zero pixels
                gray = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
                mask = (gray > 0).astype(np.uint8) * 255
                foreground_rgb = output_np
            
            # Convert foreground back to BGR for OpenCV compatibility
            foreground_bgr = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
            
            return foreground_bgr, mask
            
        except Exception as e:
            print(f"Error in background removal: {e}")
            # Return original image and empty mask as fallback
            if isinstance(input_image, np.ndarray):
                return input_image, np.zeros(input_image.shape[:2], dtype=np.uint8)
            else:
                img_array = np.array(input_image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr, np.zeros(img_array.shape[:2], dtype=np.uint8)
    
    def process_single_image(self, image_path, class_name):
        """
        Process a single image: rename, remove background, save image and mask
        
        Args:
            image_path (str): Path to input image
            class_name (str): Class name for this image
            
        Returns:
            dict: Information about processed files
        """
        try:
            # Read original image
            original_img = cv2.imread(image_path)
            if original_img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Generate new filename
            new_filename = self.generate_filename(class_name)
            base_name = os.path.splitext(new_filename)[0]
            
            # Process image - remove background and get mask
            print(f"Processing {os.path.basename(image_path)} -> {new_filename}")
            print("  Removing background using transparent-background...")
            
            # Remove background using transparent-background
            foreground_img, mask = self.remove_background(original_img)
            
            # Save original image with new name
            original_save_path = os.path.join(self.template_img_folder, new_filename)
            cv2.imwrite(original_save_path, original_img)
            
            # Save mask
            mask_save_path = os.path.join(self.template_mask_folder, new_filename)
            cv2.imwrite(mask_save_path, mask)
            
            print(f"  Saved original: {original_save_path}")
            print(f"  Saved mask: {mask_save_path}")
            
            return {
                'success': True,
                'original_path': original_save_path,
                'mask_path': mask_save_path,
                'class_name': class_name,
                'filename': new_filename
            }
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'original_file': image_path
            }
    
    def process_image_batch(self, image_paths, class_names):
        """
        Process multiple images
        
        Args:
            image_paths (list): List of image file paths
            class_names (list): List of class names corresponding to each image
            
        Returns:
            dict: Summary of processing results
        """
        if len(image_paths) != len(class_names):
            raise ValueError("Number of images must match number of class names")
        
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0,
            'classes_created': set()
        }
        
        print(f"\nProcessing {len(image_paths)} images...")
        print("-" * 50)
        
        for img_path, class_name in zip(image_paths, class_names):
            result = self.process_single_image(img_path, class_name)
            results['total_processed'] += 1
            
            if result['success']:
                results['successful'].append(result)
                results['classes_created'].add(class_name)
            else:
                results['failed'].append(result)
        
        return results
    
    def create_verification_images(self):
        """
        Create verification images showing original + mask overlay for user to verify
        """
        verification_folder = os.path.join(self.project_folder, "verification")
        os.makedirs(verification_folder, exist_ok=True)
        
        # Get all template images
        template_files = [f for f in os.listdir(self.template_img_folder) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nCreating verification images in: {verification_folder}")
        
        for filename in template_files:
            try:
                # Read original and mask
                img_path = os.path.join(self.template_img_folder, filename)
                mask_path = os.path.join(self.template_mask_folder, filename)
                
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None and mask is not None:
                    # Create colored mask overlay
                    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
                    
                    # Combine original, mask, and overlay
                    h, w = img.shape[:2]
                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    combined = np.hstack([img, mask_3ch, overlay])
                    
                    # Save verification image
                    verify_path = os.path.join(verification_folder, f"verify_{filename}")
                    cv2.imwrite(verify_path, combined)
                    
            except Exception as e:
                print(f"Error creating verification for {filename}: {e}")
        
        print(f"Verification images saved. Check {verification_folder} to verify masks are correct.")
    
    def update_config_yaml(self, config_path=None):
        """
        Update the config.yaml file with project name and extracted labels
        
        Args:
            config_path (str): Path to config.yaml file. If None, searches common locations.
        """
        # Find config.yaml file
        if config_path is None:
            possible_paths = [
                "config/config.yaml",
                os.path.join("config", "config.yaml"),
                os.path.join("..", "config", "config.yaml"),
                "config.yaml"
            ]
            
            config_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                print("Warning: config.yaml not found. Please specify the path manually.")
                return False
        
        if not os.path.exists(config_path):
            print(f"Warning: Config file not found at {config_path}")
            return False
        
        try:
            # Read current config
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Extract project name from self.project_folder (remove 'data/' prefix)
            project_name = os.path.basename(self.project_folder)
            
            # Extract unique class names from processed images
            unique_labels = self.get_unique_labels()
            
            if not unique_labels:
                print("Warning: No labels found to update config")
                return False
            
            # Update data_name
            content = re.sub(
                r'data_name:\s*["\']?[^"\'\n]*["\']?',
                f'data_name: "{project_name}"',
                content
            )
            
            # Update labels
            labels_str = '", "'.join(unique_labels)
            content = re.sub(
                r'labels:\s*\[[^\]]*\]',
                f'labels: ["{labels_str}"]',
                content
            )
            
            # Write updated config
            with open(config_path, 'w') as f:
                f.write(content)
            
            print(f"\n Updated config file: {config_path}")
            print(f"   - data_name: \"{project_name}\"")
            print(f"   - labels: {unique_labels}")
            
            return True
            
        except Exception as e:
            print(f"Error updating config file: {e}")
            return False
    
    def get_unique_labels(self):
        """
        Extract unique class names from the class_counters
        
        Returns:
            list: Sorted list of unique class names
        """
        return sorted(list(self.class_counters.keys()))
    
    def print_summary(self, results, config_updated=False):
        """Print processing summary"""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images processed: {results['total_processed']}")
        print(f"Successful: {len(results['successful'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Classes created: {sorted(list(results['classes_created']))}")
        
        if results['failed']:
            print("\nFailed images:")
            for failed in results['failed']:
                print(f"  - {failed['original_file']}: {failed['error']}")
        
        print(f"\nFiles saved to:")
        print(f"  - Original images: {self.template_img_folder}")
        print(f"  - Masks: {self.template_mask_folder}")
        
        if config_updated:
            print(f"  - Config file updated successfully")


def main():
    parser = argparse.ArgumentParser(description='Prepare template images and masks using transparent-background')
    parser.add_argument('--project_folder', type=str, required=True,
                        help='Main folder name to store all processed data')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                        help='List of image file paths to process')
    parser.add_argument('--classes', type=str, nargs='+', required=True,
                        help='List of class names corresponding to each image')
    parser.add_argument('--create_verification', action='store_true',
                        help='Create verification images to check mask quality')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config.yaml file (auto-detected if not provided)')
    parser.add_argument('--update_config', action='store_true', default=True,
                        help='Update config.yaml with project settings')
    # Arguments for background folder
    parser.add_argument('--copy_background', action='store_true', default=True,
                        help='Copy background folder to project (default: True)')
    parser.add_argument('--no_background', action='store_true',
                        help='Skip copying background folder')
    parser.add_argument('--background_path', type=str, default=None,
                        help='Path to source background folder (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.images) != len(args.classes):
        print("Error: Number of images must match number of class names")
        return
    
    # Determine whether to copy background
    copy_bg = args.copy_background and not args.no_background
    
    # Initialize preparator
    preparator = TemplatePreparator(
        args.project_folder, 
        copy_background=copy_bg,
        background_source=args.background_path
    )
    
    # Process images
    results = preparator.process_image_batch(args.images, args.classes)
    
    # Create verification images if requested
    if args.create_verification:
        preparator.create_verification_images()
    
    # Update config file
    config_updated = False
    if args.update_config:
        config_updated = preparator.update_config_yaml(args.config_path)
    
    # Print summary
    preparator.print_summary(results, config_updated)


def interactive_mode():
    """Interactive mode for easier use"""
    print("=" * 60)
    print("TEMPLATE IMAGE PREPARATION TOOL")
    print("Using transparent-background for background removal")
    print("=" * 60)
    
    # Get project folder name
    project_folder = input("Enter project folder name: ").strip()
    if not project_folder:
        print("Project folder name cannot be empty!")
        return
    
    # Ask about background folder
    print("Automatically copying background folder...")
    copy_bg = True
    background_path = None
    
    # Initialize preparator
    preparator = TemplatePreparator(
        project_folder, 
        copy_background=copy_bg,
        background_source=background_path
    )
    
    images = []
    classes = []
    
    print("\nAdd images one by one (press Enter with empty path to finish):")
    while True:
        img_path = input("Image path: ").strip()
        if not img_path:
            break
            
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            continue
            
        class_name = input("Class name for this image: ").strip()
        if not class_name:
            print("Class name cannot be empty!")
            continue
            
        images.append(img_path)
        classes.append(class_name)
        print(f"Added: {os.path.basename(img_path)} -> {class_name}")
    
    if not images:
        print("No images added!")
        return
    
    # Process images
    results = preparator.process_image_batch(images, classes)
    
    print("\nCreating verification images...")
    preparator.create_verification_images()
    
    print("Updating config.yaml file...")
    config_updated = preparator.update_config_yaml()
    
    # print summary
    preparator.print_summary(results, config_updated)


if __name__ == "__main__":
    import sys
    
    # Run interactive mode if no command line arguments
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()