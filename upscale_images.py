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

print("[INFO] loading the SR model... {}".format(args["model"]))
print("[INFO] SR Model Name : {}".format(model_name))
print("[INFO] Model Scale : {}".format(model_scale))

# init OCV's SR DNN (Deep Neural Network) object
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(args["model"])
sr.setModel(model_name, model_scale)

image = cv2.imread(args["image"])
print('[INFO] Width: {}, Height: {}'.format(image.shape[1], image.shape[0]))

# use model and time it
start_time = time.time()
upscaled_image = sr.upsample(image)
end_time = time.time()

print('[INFO] the SR Model took {:.6f} seconds'.format(end_time - start_time))
print('[INFO] Width: {}, Height: {}'.format(upscaled_image.shape[1], upscaled_image.shape[0]))

cv2.imwrite(f'examples\\{model_name}_result.png', upscaled_image)

cv2.imshow("Original : ", image)
cv2.imshow("SR : ", upscaled_image)
cv2.waitKey(0)

