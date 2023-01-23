import cv2
import os
import numpy as np

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 0, 127)


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""

    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(input_image, (left, top - 20), (left + dim[0], top + dim[1] + baseline - 20), BLACK, cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1] - 20), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()[3]
    outputs = net.forward(output_layers)
    # print(outputs[0].shape)

    return outputs

def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []

    # Rows.
    rows = outputs.shape[1]

    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                box = np.array([left, top, width, height])
                boxes.append(box)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 2 * THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image


if __name__ == '__main__':
    # Load class names.
    classesFile = "classes.txt"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Create folder for saving images
    destination_img_dir = "./yolov7_images_with_inference/"
    os.makedirs(destination_img_dir, exist_ok=True)

    # Load image.
    folder_path = "./images/"
    folder_files = os.listdir(folder_path)
    for filename in folder_files:
        frame = cv2.imread(folder_path + filename)

        # Give the weight files to the model and load the network using them.
        modelWeights = "./models/iteration_9_v7_best.onnx"
        net = cv2.dnn.readNet(modelWeights)

        # Process image.
        detections = pre_process(frame, net)
        img = post_process(frame.copy(), detections)

        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        inference = t * 1000.0 / cv2.getTickFrequency()
        label = '%.2f ms' % inference
        # print(label)

        # save inference value
        f = open(destination_img_dir + 'v7_inferences.txt', "a+")
        f.write('%.2f\n' % inference)
        f.close()

        # save detected image
        cv2.putText(img, label, (10, 15), FONT_FACE, FONT_SCALE, WHITE, THICKNESS, cv2.LINE_AA)
        cv2.imwrite(destination_img_dir + "/v7_detected_{}".format(filename), img)

    print("-----------Propeller Inspector---------------")
    print("All image files has been detected. please check the save_with_inference folder")
    # cv2.imshow('Output', img)
    cv2.waitKey(0)
