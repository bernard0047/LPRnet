import torch
import numpy as np
from utility import runner
from PIL import Image
import os
from preload import preloader

def process(img="download.jpg"):
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    model,cfg=preloader()
    with torch.no_grad():
        # for django integration
        # im = Image.open(
        #     img
        # )
        #for separate use
        im = Image.open(
            'test1.jpg'
        )
        frame = np.array(im)
        array_image, data_dictionary = runner(frame,model,cfg)
        final_image = Image.fromarray(array_image, "RGB")
        print(data_dictionary[0][3])
        # final_image.save(f"media/OUT/{img.name}") #for django integration
        final_image.save(f"{current_path}/{img}")  # for Separate Use
    # return data_dictionary , img.name
    return data_dictionary, img


if __name__ == "__main__":
    process()