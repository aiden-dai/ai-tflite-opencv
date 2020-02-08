"""
Run Pose detection on images, Press ESC to exit the program
For Raspberry PI, please use `import tflite_runtime.interpreter as tflite` instead
"""
import re
import cv2
import numpy as np

import tensorflow.lite as tflite
# import tflite_runtime.interpreter as tflite

from PIL import Image


class Part:
    r"""Enum of Detected Part IDs, for example, 0 is Nose"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


def sigmoid(x):
    return 1.0 / (1.0 + 1.0 / np.exp(x))


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index):
    r"""Process an image, Return a list of positions in a 4-Tuple (pos_x, pos_y, offset_x, offset_y)"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim
    input_data = (np.float32(input_data) - 127.5) / 127.5  # float point

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    # print(output_details)

    output_data = np.squeeze(
        interpreter.get_tensor(output_details[0]['index']))
    offset_data = np.squeeze(
        interpreter.get_tensor(output_details[1]['index']))

    points = []

    total_row, total_col, total_points = output_data.shape

    # totally 17 points
    for k in range(total_points):
        max_score = output_data[0][0][k]
        max_row = 0
        max_col = 0
        for row in range(total_row):
            for col in range(total_col):
                if (output_data[row][col][k] > max_score):
                    max_score = output_data[row][col][k]
                    max_row = row
                    max_col = col

        points.append((max_row, max_col))
        # print(sigmoid(max_score))

    positions = []

    for idx, point in enumerate(points):
        pos_y, pos_x = point

        # y is row, x is column
        offset_x = offset_data[pos_y][pos_x][idx + 17]
        offset_y = offset_data[pos_y][pos_x][idx]

        positions.append((pos_x, pos_y, offset_x, offset_y))
        # confidenceScores = sigmoid(output_data[pos_y][pos_x][idx])
        # print('confidenceScores {}'.format(confidenceScores))

    return positions


def display_result(positions, frame):
    r"""Display Detected Points in circles"""
    size = 5
    color = (255, 0, 0)  # Blue color
    thickness = 3

    width = frame.shape[1]
    height = frame.shape[0]

    for pos in positions:
        pos_x, pos_y, offset_x, offset_y = pos

        # Calculating the x and y coordinates
        x = int(pos_x / 8 * width + offset_x)
        y = int(pos_y / 8 * height + offset_y)

        cv2.circle(frame, (x, y), size, color, thickness)

    cv2.imshow('Pose Detection', frame)


if __name__ == "__main__":

    model_path = 'data/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite'
    image_path = 'data/person.jpg'

    interpreter = load_model(model_path)

    input_details = interpreter.get_input_details()
    # Get Width and Height
    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]

    # Get input index
    input_index = input_details[0]['index']

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(frame.shape)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((width, height))

    positions = process_image(interpreter, image, input_index)
    display_result(positions, frame)

    key = cv2.waitKey(0)
    if key == 27:  # esc
        cv2.destroyAllWindows()
