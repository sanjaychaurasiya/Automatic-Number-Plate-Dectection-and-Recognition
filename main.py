import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import util


def number_plate_reader(img):
    """Function to detecting number plate and reading text on them."""

    # define constants
    model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
    model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
    class_names_path = os.path.join('.', 'model', 'classes.names')

    # load class names
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image
    img = cv2.imread(image_path)

    H, W, _ = img.shape

    # convert image, The input to the network is a so-called blob object
    blob = cv2.dnn.blobFromImage(img, 1/255., (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    """Non-Max Supression (NMS) is a technique used to select one bounding box for an object if multiple bounding boxes 
    were detected with varying probability scores by object detection algorithms(example: Faster R-CNN,YOLO)"""
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)


    # plot
    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        licence_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        upscaled = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            4)
        licence_plate_gray = cv2.cvtColor(licence_plate, cv2.COLOR_BGR2GRAY)

        # Save the image in PNG format
        cv2.imwrite("./saved_image.png", licence_plate_gray)

        """ Trocr model from transformers library """
        model_path = './model/Trocr-large-printed'

        pipe = pipeline("image-to-text", model=model_path)
        generated_ids = pipe("./saved_image.png")

        # Use this code to view the images, remove this while integrating.
        # plt.figure()
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #
        # plt.figure()
        # plt.imshow(cv2.cvtColor(licence_plate, cv2.COLOR_BGR2RGB))
        #
        # plt.figure()
        # plt.imshow(cv2.cvtColor(licence_plate_gray, cv2.COLOR_BGR2RGB))
        # plt.show()

        return generated_ids


if __name__ == "__main__":
    image_path = "./data/high_quality/car.jpg"
    print(f"Text on the Number Plate: {number_plate_reader(image_path)[0]['generated_text']}")
