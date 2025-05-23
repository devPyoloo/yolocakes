''' Usage:
In the main script, import "read_all_args" function
In this script, modify parameters only "set_fixed_arguments" function
'''

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Set fixed arguments that doesn't need to change
def set_fixed_arguments(args):
    # Use os.path.join() for all paths
    args.f_data_src = os.path.join(ROOT, "data", args.data_name) + os.sep
    args.f_data_dst = os.path.join(ROOT, "data", args.data_name + "_generated") + os.sep
    args.f_data_eval = os.path.join(ROOT, "data", args.data_name + "_eval") + os.sep
    
    args.f_template_img = os.path.join(args.f_data_src, "template_img") + os.sep
    args.f_template_mask = os.path.join(args.f_data_src, "template_mask") + os.sep
    args.f_background = os.path.join(args.f_data_src, "background") + os.sep
    
    args.f_yolo_images = os.path.join(args.f_data_dst, "train", "images") + os.sep
    args.f_yolo_labels = os.path.join(args.f_data_dst, "train", "labels") + os.sep
    args.f_yolo_images_with_bbox = os.path.join(args.f_data_dst, "images_with_bbox") + os.sep
    
    # args.f_yolo_classes = os.path.join(args.f_data_dst, "classes.names")
    args.f_yolo_classes = os.path.join(args.f_data_dst, "classes.txt")
    args.f_yolo_train = os.path.join(args.f_data_dst, "train.txt")
    args.f_yolo_valid = os.path.join(args.f_data_dst, "valid.txt")
    args.f_yolo_valid_images = os.path.join(args.f_data_dst, "valid_images") + os.sep
    args.f_yolo_config = os.path.join(args.f_data_dst, "yolo.cfg")
    args.f_yolo_data = os.path.join(args.f_data_dst, "yolo.data")
    
    print("\n--- Path Debug ---")
    print(f"ROOT: {os.path.abspath(ROOT)}")
    print(f"Template images: {os.path.abspath(args.f_template_img)}")
    print(f"First template exists: {os.path.exists(os.path.join(args.f_template_img, 'bottle_1.png'))}")
# --------------------------------------------------------------------
# --------------------------------------------------------------------


if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.join(ROOT, "..")  # Go up one level
    sys.path.append(ROOT)
    
import utils.lib_common_funcs as cf

def read_all_args(config_file="config/config.yaml"):
    
    # Read args from yaml file
    args_dict = cf.load_yaml(config_file)
    args = cf.dict2class(args_dict)
    
    # Some fixed arguments that doesn't need change
    set_fixed_arguments(args)
    
    return args 
