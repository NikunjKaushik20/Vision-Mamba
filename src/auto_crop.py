import cv2
import numpy as np
import os
import glob
from pathlib import Path

def auto_crop_xray(image_path, output_path=None):
    """
    Attempts to identify the bright X-ray box inside a photograph
    (e.g. taken of a monitor or holding up to light) and crops to it.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    orig_img = img.copy()
    
    # 1. Convert to grayscale and blur to remove noise (like Moiré)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # 2. Thresholding to find the bright monitor/screen area
    # Otsu's thresholding usually works well for dark room + bright screen
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optional morphological operations to close holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return orig_img # fallback if nothing found
        
    # 4. Assume the largest contour by area is the X-ray screen
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    img_area = img.shape[0] * img.shape[1]
    
    # If the largest bright area is tiny, it failed
    if area < 0.1 * img_area:
        return orig_img
        
    # 5. Get bounding box and crop
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Optional: add a small padding (margin)
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2*margin)
    h = min(img.shape[0] - y, h + 2*margin)
    
    cropped = orig_img[y:y+h, x:x+w]
    
    if output_path is not None:
        cv2.imwrite(output_path, cropped)
        
    return cropped

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(input_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(input_dir, "*.[jJ][pP][eE][gG]"))
                  
    print(f"Found {len(image_paths)} images to crop")
    
    for path in image_paths:
        filename = os.path.basename(path)
        out_path = os.path.join(output_dir, filename)
        
        print(f"Cropping {filename}...")
        auto_crop_xray(path, out_path)
        
if __name__ == "__main__":
    import shutil
    
    in_dir = "d:/Vision-Mamba/real_life_test"
    out_dir = "d:/Vision-Mamba/real_life_test_cropped"
    
    process_directory(in_dir, out_dir)
    print("Done! Check real_life_test_cropped for results.")
