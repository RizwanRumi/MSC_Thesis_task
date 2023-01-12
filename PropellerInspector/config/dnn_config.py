import sys
import cv2 as cv

class ModelConfiguration:
    def __init__(self, model):
        self.model = model

    def modelConfig(self, dnn_name):
        if dnn_name == "YOLOV5":
            network = cv.dnn.readNetFromONNX(self.model)
        else:
            network = None

        cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"
        # network configuration with or without CUDA
        if cuda:
            print("Attempt to use CUDA")
            network.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
            network.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Running on CPU")
            network.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            network.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        return network
