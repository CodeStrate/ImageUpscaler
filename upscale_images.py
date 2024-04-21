import argparse # for passing arguments while running the file
import time, cv2, os


# Make the parser
ap = argparse.ArgumentParser()

ap.add_argument('-m', '--model', required=True, help='Path to the Upscaling model to be used.')
ap.add_argument('-i', '--image', required=True, help='Path to the input image.')

args = vars(ap.parse_args())

# extract scale and model name from the path
model_name = args["model"].split(os.path.sep)[-1].split("_")[0].lower()
model_scale = args["model"].split("_x")[-1]
model_scale = int(model_scale[:model_scale.find(".")])