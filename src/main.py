"""
----------------------------------ABOUT-----------------------------------
Author: Arun Baskaran
--------------------------------------------------------------------------
"""

import lib_imports
from aux_funcs import *
import model_params


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: main <mode>", file=sys.stderr)
        sys.exit(-1)
    mode = sys.argv[1] 
    train_images, train_labels, test_images, test_labels, validation_images, validation_labels = load_images_labels()
    
    if mode == "training" :
        model = train_model()
    
    elif mode =="load":
        model = load_model()
        
    test_accuracy(model)
    
    y_classes = get_predicted_classes(model)
    
    feature_segmentation()
    



