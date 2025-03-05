import cv2
import numpy as np
import imutils

def preprocess_photo(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image. Check the path.")
        exit()
    
    image = imutils.resize(image, height=1280)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 1)
    
    adaptive_thresh = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    edges_detected = cv2.Canny(adaptive_thresh, 50, 150)
    return image, grayscale, edges_detected

def find_document_edges(edges_detected):
    contours, _ = cv2.findContours(edges_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_surface = 0
    best_shape = None
    
    for shape in contours:
        surface_area = cv2.contourArea(shape)
        if surface_area > 1000:
            perimeter = cv2.arcLength(shape, True)
            approx_shape = cv2.approxPolyDP(shape, 0.02 * perimeter, True)
            if len(approx_shape) == 4 and surface_area > max_surface:
                best_shape = approx_shape
                max_surface = surface_area
    
    return best_shape

def apply_perspective_transform(original_image, shape):
    if shape is None:
        print("Error: No document found in the image.")
        exit()
    
    shape = shape.reshape((4, 2))
    sorted_corners = np.zeros((4, 2), dtype="float32")
    
    sum_corners = shape.sum(axis=1)
    sorted_corners[0] = shape[np.argmin(sum_corners)]
    sorted_corners[2] = shape[np.argmax(sum_corners)]
    
    diff_corners = np.diff(shape, axis=1)
    sorted_corners[1] = shape[np.argmin(diff_corners)]
    sorted_corners[3] = shape[np.argmax(diff_corners)]
    
    (top_left, top_right, bottom_right, bottom_left) = sorted_corners
    doc_width = max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))
    doc_height = max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))
    
    destination_corners = np.array([
        [0, 0],
        [doc_width - 1, 0],
        [doc_width - 1, doc_height - 1],
        [0, doc_height - 1]
    ], dtype="float32")
    
    transformation_matrix = cv2.getPerspectiveTransform(sorted_corners, destination_corners)
    corrected_image = cv2.warpPerspective(original_image, transformation_matrix, (int(doc_width), int(doc_height)))
    
    if corrected_image.shape[0] < corrected_image.shape[1]:
        corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_90_CLOCKWISE)
    
    return corrected_image

# Execution Pipeline
photo_path = input("Enter image path: ")
original_photo, grayscale_photo, detected_edges = preprocess_photo(photo_path)
document_shape = find_document_edges(detected_edges)
scanned_document = apply_perspective_transform(original_photo, document_shape)

# Save and Display
cv2.imshow("Scanned Document", scanned_document)
cv2.imwrite("scanned_document.jpg", scanned_document)
print("Scanned document saved as 'scanned_document.jpg'")
cv2.waitKey(0)
cv2.destroyAllWindows()