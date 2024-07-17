import os
import json
import random
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp

def measure(path_to_warped_image_mini, lines):
    heart_thres_x = 0
    head_thres_x = 0
    life_thres_y = 0

    # Load content from JSON file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'palm_reading_content.json')
    with open(json_path, 'r') as file:
        palm_reading_content = json.load(file)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(path_to_warped_image_mini), 1)
        image_height, image_width, _ = image.shape

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        hand_landmarks = results.multi_hand_landmarks[0]

        zero = hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y
        one = hand_landmarks.landmark[mp_hands.HandLandmark(1).value].y
        five = hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x
        nine = hand_landmarks.landmark[mp_hands.HandLandmark(9).value].x
        thirteen = hand_landmarks.landmark[mp_hands.HandLandmark(13).value].x

        heart_thres_x = image_width * (1 - (nine + (five - nine) * 2 / 5))
        head_thres_x = image_width * (1 - (thirteen + (nine - thirteen) / 3))
        life_thres_y = image_height * (one + (zero - one) / 3)

    im = Image.open(path_to_warped_image_mini)
    width = 3
    if (None in lines) or (len(lines) < 3):
        return None, None
    else:
        draw = ImageDraw.Draw(im)

        heart_line = lines[0]
        head_line = lines[1]
        life_line = lines[2]

        heart_line_points = [tuple(reversed(l[:2])) for l in heart_line]
        heart_line_tip = heart_line_points[0]
        if heart_line_tip[0] < heart_thres_x:
            heart_content_2 = random.choice(list(palm_reading_content['heart']['long'].values()))
            marriage_content_2 = random.choice(list(palm_reading_content['marriage']['long'].values()))
        else:
            heart_content_2 = random.choice(list(palm_reading_content['heart']['short'].values()))
            marriage_content_2 = random.choice(list(palm_reading_content['marriage']['short'].values()))
        draw.line(heart_line_points, fill="red", width=width)

        head_line_points = [tuple(reversed(l[:2])) for l in head_line]
        head_line_tip = head_line_points[-1]
        if head_line_tip[0] > head_thres_x:
            head_content_2 = random.choice(list(palm_reading_content['head']['long'].values()))
            fate_content_2 = random.choice(list(palm_reading_content['fate']['long'].values()))
        else:
            head_content_2 = random.choice(list(palm_reading_content['head']['short'].values()))
            fate_content_2 = random.choice(list(palm_reading_content['fate']['short'].values()))
        draw.line(head_line_points, fill="green", width=width)

        life_line_points = [tuple(reversed(l[:2])) for l in life_line]
        life_line_tip = life_line_points[-1]
        if life_line_tip[1] > life_thres_y:
            life_content_2 = random.choice(list(palm_reading_content['life']['long'].values()))
        else:
            life_content_2 = random.choice(list(palm_reading_content['life']['short'].values()))
        draw.line(life_line_points, fill="blue", width=width)

        contents = [heart_content_2, head_content_2, life_content_2, marriage_content_2, fate_content_2]
        return im, contents
