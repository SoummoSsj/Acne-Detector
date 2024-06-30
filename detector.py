import cv2
import numpy as np

# Load the cascades
face_cascade = cv2.CascadeClassifier('frontal_face.xml')
eye_cascade = cv2.CascadeClassifier('frontal_eye.xml')
mouth_cascade = cv2.CascadeClassifier('mouth.xml')


def detect_face(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=(100, 100))
    return faces

def detect_eye(image):
    eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30))
    return eyes

def detect_mouth(image):
    mouths = mouth_cascade.detectMultiScale(image, scaleFactor=1.5, minNeighbors=11, minSize=(30, 30))
    return mouths

def detect_acne(face_region):
    # Convert face region to grayscale
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to the grayscale image
    blurred_face = cv2.GaussianBlur(gray_face, (15, 15), 0)
    
    # Subtract the blurred image from the original grayscale image
    acne_mask = cv2.absdiff(gray_face, blurred_face)
    
    # Threshold the difference image to create a binary mask of potential acne regions
    _, acne_mask = cv2.threshold(acne_mask, 20, 255, cv2.THRESH_BINARY)
    
    # Find contours to filter out large areas (like hair)
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_acne_mask = np.zeros_like(acne_mask)
    for contour in contours:
        if cv2.contourArea(contour) < 5:  # Adjust the area threshold based on your needs
            cv2.drawContours(filtered_acne_mask, [contour], -1, 255, -1)
    
    # Dilate the mask to include surrounding areas of the detected acne
    kernel = np.ones((3, 3), np.uint8)
    filtered_acne_mask = cv2.dilate(filtered_acne_mask, kernel, iterations=2)
    
    return filtered_acne_mask

def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save and display original image
    cv2.imwrite('original_image.jpg', image)
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    # Detect faces
    faces = detect_face(image)

    for i, (x, y, w, h) in enumerate(faces):
        # Extract face region
        face_region = image[y:y+h, x:x+w]
        gray_face_region = gray[y:y+h, x:x+w]

        # Save and display face region
        cv2.imwrite(f'face_region_{i}.jpg', face_region)
        cv2.imshow(f'Face Region {i}', face_region)
        cv2.waitKey(0)

        # Detect eyes within the face region
        eyes = detect_eye(face_region)
        eye_regions = []
        for j, (ex, ey, ew, eh) in enumerate(eyes):
            eye_regions.append((ex, ey, ew, eh, face_region[ey:ey+eh, ex:ex+ew].copy()))
            face_region[ey:ey+eh, ex:ex+ew] = cv2.cvtColor(gray_face_region[ey:ey+eh, ex:ex+ew], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f'eye_region_{i}_{j}.jpg', face_region[ey:ey+eh, ex:ex+ew])

        # Save and display face region with grayscaled eyes
        cv2.imwrite(f'face_region_grayscaled_eyes_{i}.jpg', face_region)
        cv2.imshow(f'Face Region with Grayscaled Eyes {i}', face_region)
        cv2.waitKey(0)

        # Detect mouth within the face region
        mouths = detect_mouth(face_region)
        mouth_regions = []
        for k, (mx, my, mw, mh) in enumerate(mouths):
            mouth_regions.append((mx, my, mw, mh, face_region[my:my+mh, mx:mx+mw].copy()))
            face_region[my:my+mh, mx:mx+mw] = cv2.cvtColor(gray_face_region[my:my+mh, mx:mx+mw], cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f'mouth_region_{i}_{k}.jpg', face_region[my:my+mh, mx:mx+mw])

        # Save and display face region with grayscaled eyes and mouth
        cv2.imwrite(f'face_region_grayscaled_eyes_mouth_{i}.jpg', face_region)
        cv2.imshow(f'Face Region with Grayscaled Eyes and Mouth {i}', face_region)
        cv2.waitKey(0)

        # Create a mask for the eyes and mouth
        mask = np.ones(face_region.shape[:2], dtype=np.uint8) * 255
        for (ex, ey, ew, eh, _) in eye_regions:
            mask[ey:ey+eh, ex:ex+ew] = 0
        for (mx, my, mw, mh, _) in mouth_regions:
            mask[my:my+mh, mx:mx+mw] = 0

        # Detect acne in the face region
        acne_mask = detect_acne(face_region)

        # Save and display acne mask
        cv2.imwrite(f'acne_mask_{i}.jpg', acne_mask)
        cv2.imshow(f'Acne Mask {i}', acne_mask)
        cv2.waitKey(0)

        # Combine the acne mask with the eyes and mouth mask
        combined_mask = cv2.bitwise_and(acne_mask, acne_mask, mask=mask)

        # Save and display combined mask
        cv2.imwrite(f'combined_mask_{i}.jpg', combined_mask)
        cv2.imshow(f'Combined Mask {i}', combined_mask)
        cv2.waitKey(0)

        # Inpaint the detected acne regions
        inpainted_face = cv2.inpaint(face_region, combined_mask, 3, cv2.INPAINT_TELEA)

        # Save and display inpainted face
        cv2.imwrite(f'inpainted_face_{i}.jpg', inpainted_face)
        cv2.imshow(f'Inpainted Face {i}', inpainted_face)
        cv2.waitKey(0)

        # Restore the original color of the eyes and mouth
        for (ex, ey, ew, eh, eye_region) in eye_regions:
            inpainted_face[ey:ey+eh, ex:ex+ew] = eye_region
        for (mx, my, mw, mh, mouth_region) in mouth_regions:
            inpainted_face[my:my+mh, mx:mx+mw] = mouth_region

        # Save and display restored face region
        cv2.imwrite(f'restored_face_region_{i}.jpg', inpainted_face)
        cv2.imshow(f'Restored Face Region {i}', inpainted_face)
        cv2.waitKey(0)

        # Place the inpainted face region back into the image
        image[y:y+h, x:x+w] = inpainted_face

    # Save and display the final result
    cv2.imwrite('final_processed_image.jpg', image)
    cv2.imshow('Final Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
process_image('images.jpg')
