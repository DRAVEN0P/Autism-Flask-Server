# import numpy as np
# import cv2
# import os
# import pandas as pd
# import dlib

# cnn_face_detector_model_path = './models/mmod_human_face_detector.dat'
# shape_predictor_model_path = './models/shape_predictor_68_face_landmarks.dat'

# cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model_path)
# predictor = dlib.shape_predictor(shape_predictor_model_path)

# def detect_face(gray_image):
#     faces = cnn_face_detector(gray_image, 1)
#     if len(faces) == 0:
#         print("No faces detected.")
#         return None, None
#     face = faces[0].rect
#     return face, gray_image

# def get_landmarks(gray_image, face):
#     shape = predictor(gray_image, face)
#     landmarks = np.array([[p.x, p.y] for p in shape.parts()])
#     return landmarks

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error reading image: {image_path}")
#         return None
#     resized_image = cv2.resize(image, (256, 256))
#     gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#     return resized_image, gray_image

# def calculate_distance(point1, point2):
#     return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# def extract_features(landmarks):
#     feature_definitions = {
#         'Eye_Outer_Width': (36, 45),
#         'Left_Eye_Width': (36, 39),
#         'Right_Eye_Width': (42, 45),
#         'Nose_Width': (31, 35),
#         'Mouth_Width': (48, 54),
#         'Nose_Height': (27, 33),
#         'Cheek_Height_Left': (36, 2)
#     }
    
#     features = {}
#     for feature_name, (start, end) in feature_definitions.items():
#         start_point = landmarks[start]
#         end_point = landmarks[end]
#         if feature_name == 'Cheek_Height_Left':
#             features[feature_name] = calculate_distance(start_point, landmarks[end])
#         else:
#             features[feature_name] = calculate_distance(start_point, end_point)
    
#     return features

# def process_image(image_path):
#     image, gray_image = preprocess_image(image_path)
#     if image is None:
#         return None
#     face, gray_image = detect_face(gray_image)
#     if face is None:
#         return None
#     landmarks = get_landmarks(gray_image, face)
#     features = extract_features(landmarks)
#     return features

# def print_single_image_features(image_path):
#     features = process_image(image_path)
#     if features:
#         # Print features in the desired format
#         print("Feature                    Value")
#         return{
#             list(features.items())[0][0]:list(features.items())[0][1],
#             list(features.items())[1][0]:list(features.items())[1][1],
#             list(features.items())[2][0]:list(features.items())[2][1],
#             list(features.items())[3][0]:list(features.items())[3][1],
#             list(features.items())[4][0]:list(features.items())[4][1],
#             list(features.items())[5][0]:list(features.items())[5][1],
#             list(features.items())[6][0]:list(features.items())[6][1],
#         }
# def load_features_from_directory(directory):
#     features = []
#     labels = []
    
#     for label in os.listdir(directory):
#         label_dir = os.path.join(directory, label)
#         if not os.path.isdir(label_dir):
#             continue
#         for file_name in os.listdir(label_dir):
#             if file_name.endswith('.npy'):
#                 file_path = os.path.join(label_dir, file_name)
#                 feature_data = np.load(file_path)
#                 features.append(feature_data)
#                 labels.append(label)

#     return np.vstack(features), np.array(labels)
# def ref_table():
    
#     data_dir = './output_imag'

#     # Load features and labels
#     X, y = load_features_from_directory(data_dir)

#     # Create DataFrame
#     df = pd.DataFrame(X, columns=[
#     'Eye_Outer_Width', 'Left_Eye_Width', 'Right_Eye_Width', 'Nose_Width',
#     'Mouth_Width', 'Nose_Height', 'Cheek_Height_Left'
#     ])

#     df['Class'] = y

#     # Compute mean and standard deviation by class
#     mean_std_by_class = df.groupby('Class').agg(['mean', 'std'])

#     # Extract class names
#     class_names = mean_std_by_class.index

#     # Initialize a dictionary to hold the formatted results
#     formatted_results = {}

#     # Format results
#     for feature in mean_std_by_class.columns.levels[0]:
#         means = [mean_std_by_class[(feature, 'mean')][cls] for cls in class_names]
#         stds = [mean_std_by_class[(feature, 'std')][cls] for cls in class_names]
#         formatted_results[feature] = [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(means, stds)]

#     # Print the results
#     print("Reference Feature Statistics Table")
#     print(f"{'Feature':<25} {' '.join(class_names)}")

#     return{
#         list(formatted_results.items())[0][0]:list(formatted_results.items())[0][1],
#         list(formatted_results.items())[1][0]:list(formatted_results.items())[1][1],
#         list(formatted_results.items())[2][0]:list(formatted_results.items())[2][1],
#         list(formatted_results.items())[3][0]:list(formatted_results.items())[3][1],
#         list(formatted_results.items())[4][0]:list(formatted_results.items())[4][1],
#         list(formatted_results.items())[5][0]:list(formatted_results.items())[5][1],
#         list(formatted_results.items())[6][0]:list(formatted_results.items())[6][1],
#     }




# # Predict on a new image
# def predict_image(model, scaler, image_path, target_size=(64, 64)):
#     img_features = print_single_image_features(image_path)
#     image = cv2.imread(image_path)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     if image is not None:
#         image = cv2.resize(image, target_size)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#         faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         if len(faces)>0:
#             image = image / 255.0  # Normalize
#             image = image.flatten().reshape(1, -1)
#             image = scaler.transform(image)
#             probabilities = model.predict_proba(image)
#             prediction = model.predict(image)
            
#             print(f'\nPredicted Class: {prediction[0]}')
#             print(f'Confidence for class 0: {probabilities[0][0] * 100:.2f}%')
#             print(f'Confidence for class 1: {probabilities[0][1] * 100:.2f}%\n')
#             return {
#                 'predicted_class': prediction[0],
#                 'confidence_class_0': probabilities[0][0] * 100,
#                 'confidence_class_1': probabilities[0][1] * 100,
#                 'image_features': img_features
#             }
#         else:
#             #Here you need to code for retaking the image
#             return {
#                 "confidence_class_0": 0.00,
#                 "confidence_class_1": 0.00,
#                 "predicted_class": "None",
#                 "image_features": {
#                     "Cheek_Height_Left": 0.0,
#                     "Eye_Outer_Width": 0.0,
#                     "Left_Eye_Width": 0.0,
#                     "Mouth_Width": 0.0,
#                     "Nose_Height": 0.0,
#                     "Nose_Width": 0.0,
#                     "Right_Eye_Width": 0.0
#                 },
#         }
import numpy as np
import cv2
import os
import pandas as pd
import dlib

# Paths to models
cnn_face_detector_model_path = 'models/mmod_human_face_detector.dat'
shape_predictor_model_path = 'models/shape_predictor_68_face_landmarks.dat'

# Load dlib models
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_model_path)
predictor = dlib.shape_predictor(shape_predictor_model_path)

def detect_face(gray_image):
    if gray_image is None:
        print("Error: Image is None.")
        return None, None
    
    # Check if the image is 8-bit grayscale
    if gray_image.dtype != np.uint8:
        print(f"Error: Expected 8-bit grayscale image, got {gray_image.dtype}.")
        return None, None
    
    faces = cnn_face_detector(gray_image, 1)
    if len(faces) == 0:
        print("No faces detected.")
        return None, None
    
    face = faces[0].rect
    return face, gray_image

def get_landmarks(gray_image, face):
    shape = predictor(gray_image, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return None, None
    
    resized_image = cv2.resize(image, (256, 256))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Debug information
    print(f"Preprocessed image shape: {resized_image.shape}")
    print(f"Gray image dtype: {gray_image.dtype}, shape: {gray_image.shape}")
    
    return resized_image, gray_image

def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def extract_features(landmarks):
    feature_definitions = {
        'Eye_Outer_Width': (36, 45),
        'Left_Eye_Width': (36, 39),
        'Right_Eye_Width': (42, 45),
        'Nose_Width': (31, 35),
        'Mouth_Width': (48, 54),
        'Nose_Height': (27, 33),
        'Cheek_Height_Left': (36, 2)
    }
    
    features = {}
    for feature_name, (start, end) in feature_definitions.items():
        start_point = landmarks[start]
        end_point = landmarks[end]
        if feature_name == 'Cheek_Height_Left':
            features[feature_name] = calculate_distance(start_point, landmarks[end])
        else:
            features[feature_name] = calculate_distance(start_point, end_point)
    
    return features

def process_image(image_path):
    image, gray_image = preprocess_image(image_path)
    if image is None or gray_image is None:
        return None
    face, gray_image = detect_face(gray_image)
    if face is None:
        return None
    landmarks = get_landmarks(gray_image, face)
    features = extract_features(landmarks)
    return features

def print_single_image_features(image_path):
    features = process_image(image_path)
    if features:
        # Print features in the desired format
        print("Feature                    Value")
        return {
            feature_name: value
            for feature_name, value in features.items()
        }
    return None

def load_features_from_directory(directory):
    features = []
    labels = []
    
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(label_dir, file_name)
                feature_data = np.load(file_path)
                features.append(feature_data)
                labels.append(label)

    return np.vstack(features), np.array(labels)

def ref_table():
    data_dir = './output_imag'
    
    # Load features and labels
    X, y = load_features_from_directory(data_dir)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[
        'Eye_Outer_Width', 'Left_Eye_Width', 'Right_Eye_Width', 'Nose_Width',
        'Mouth_Width', 'Nose_Height', 'Cheek_Height_Left'
    ])
    
    df['Class'] = y

    # Compute mean and standard deviation by class
    mean_std_by_class = df.groupby('Class').agg(['mean', 'std'])

    # Extract class names
    class_names = mean_std_by_class.index

    # Initialize a dictionary to hold the formatted results
    formatted_results = {}

    # Format results
    for feature in mean_std_by_class.columns.levels[0]:
        means = [mean_std_by_class[(feature, 'mean')][cls] for cls in class_names]
        stds = [mean_std_by_class[(feature, 'std')][cls] for cls in class_names]
        formatted_results[feature] = [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(means, stds)]

    # Print the results
    print("Reference Feature Statistics Table")
    print(f"{'Feature':<25} {' '.join(class_names)}")

    return {
        feature_name: values
        for feature_name, values in formatted_results.items()
    }

def predict_image(model, scaler, image_path, target_size=(64, 64)):
    img_features = print_single_image_features(image_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return {
            "confidence_class_0": 0.00,
            "confidence_class_1": 0.00,
            "predicted_class": "None",
            "image_features": img_features or {}
        }

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.resize(image, target_size)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        image_gray = image_gray / 255.0  # Normalize
        image_gray = image_gray.flatten().reshape(1, -1)
        image_gray = scaler.transform(image_gray)
        probabilities = model.predict_proba(image_gray)
        prediction = model.predict(image_gray)
        
        print(f'\nPredicted Class: {prediction[0]}')
        print(f'Confidence for class 0: {probabilities[0][0] * 100:.2f}%')
        print(f'Confidence for class 1: {probabilities[0][1] * 100:.2f}%\n')
        
        return {
            'predicted_class': prediction[0],
            'confidence_class_0': probabilities[0][0] * 100,
            'confidence_class_1': probabilities[0][1] * 100,
            'image_features': img_features
        }
    else:
        return {
            "confidence_class_0": 0.00,
            "confidence_class_1": 0.00,
            "predicted_class": "None",
            "image_features": img_features or {
                "Cheek_Height_Left": 0.0,
                "Eye_Outer_Width": 0.0,
                "Left_Eye_Width": 0.0,
                "Mouth_Width": 0.0,
                "Nose_Height": 0.0,
                "Nose_Width": 0.0,
                "Right_Eye_Width": 0.0
            },
        }
