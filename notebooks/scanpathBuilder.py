import cv2
import numpy as np
import os
import json
from pathlib import Path

# Global variables for mouse callback
points = []  # List to store (x, y, duration, type) fixation points
img = None   # Current image
img_backup = None  # Backup for resetting visualization

def mouse_callback(event, x, y, flags, param):
    global points, img, img_backup
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click
        # Check for modifier keys
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            point_type = "hardToRead"
            color = (0, 255, 255)  # Yellow for hardToRead
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:
            point_type = "Fatigue"
            color = (0, 255, 0)  # Green for Fatigue
        else:
            point_type = "Good"
            color = (0, 0, 255)  # Red for Good
        points.append((x, y, 200, point_type))
        cv2.circle(img, (x, y), 5, color, -1)
        if len(points) > 1:
            # Determine line color based on point types
            prev_type = points[-2][3]
            curr_type = point_type
            line_color = (0, 0, 255) if prev_type == "Fatigue" and curr_type == "Fatigue" else \
                         (0, 255, 255) if prev_type == "hardToRead" and curr_type == "hardToRead" else (255, 0, 0)
            cv2.line(img, points[-2][:2], points[-1][:2], line_color, 2)  # Red for Fatigue-Fatigue, Yellow for hardToRead-hardToRead, Blue otherwise
        cv2.imshow('Image', img)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to undo last point
        if points:
            points.pop()
            img = img_backup.copy()  # Reset to backup
            # Redraw remaining points and lines
            for i, pt in enumerate(points):
                color = (0, 0, 255) if pt[3] == "Good" else (0, 255, 0) if pt[3] == "Fatigue" else (0, 255, 255)
                cv2.circle(img, pt[:2], 5, color, -1)
                if i > 0:
                    prev_type = points[i-1][3]
                    curr_type = pt[3]
                    line_color = (0, 0, 255) if prev_type == "Fatigue" and curr_type == "Fatigue" else \
                                 (0, 255, 255) if prev_type == "hardToRead" and curr_type == "hardToRead" else (255, 0, 0)
                    cv2.line(img, points[i-1][:2], pt[:2], line_color, 2)
            cv2.imshow('Image', img)

def main(image_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    scanpaths = {}  # Dictionary to store all scanpaths
    
    for idx, filename in enumerate(image_files):
        global points, img, img_backup
        
        # Initialize variations list for this image
        image_variations = []
        
        # Load image once
        img_path = os.path.join(image_dir, filename)
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Error loading {filename}. Skipping.")
            continue
        
        # Get scale factor for coordinate conversion
        scale_factor = 1.0
        if original_img.shape[1] > 1600 or original_img.shape[0] > 900:
            scale_factor = min(1600 / original_img.shape[1], 900 / original_img.shape[0])
        
        print(f"\nAnnotating {filename} ({idx+1}/{len(image_files)})")
        print(f"You need to create 5 variations for this image.")

        # Collect 5 variations for this image
        for variation in range(1, 6):
            points = []  # Reset points for each variation
            
            # Prepare image for display
            if scale_factor < 1.0:
                img = cv2.resize(original_img, None, fx=scale_factor, fy=scale_factor)
            else:
                img = original_img.copy()
            img_backup = img.copy()
            
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Image', img.shape[1], img.shape[0])
            cv2.setMouseCallback('Image', mouse_callback)
            cv2.imshow('Image', img)
            
            print(f"\nVariation {variation}/7 for {filename}")
            print("Controls: Left click: 'Good' | Shift+Left: 'Fatigue' | Ctrl+Left: 'hardToRead'")
            print("Right click: undo | 'n': save variation | 'r': reset | 's': skip this variation | 'q': quit")
            
            variation_saved = False
            while not variation_saved:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n'):  # Save current variation
                    if len(points) > 0:
                        # Convert coordinates back to original image scale
                        scaled_points = []
                        for x, y, duration, point_type in points:
                            orig_x = int(x / scale_factor) if scale_factor < 1.0 else x
                            orig_y = int(y / scale_factor) if scale_factor < 1.0 else y
                            scaled_points.append([orig_x, orig_y, duration, point_type])
                        
                        image_variations.append(scaled_points)
                        print(f"Saved variation {variation}: {len(points)} points")
                        variation_saved = True
                    else:
                        print("No points to save! Add some points first.")
                        
                elif key == ord('r'):  # Reset current variation
                    points = []
                    img = img_backup.copy()
                    cv2.imshow('Image', img)
                    
                elif key == ord('s'):  # Skip this variation
                    print(f"Skipped variation {variation}")
                    image_variations.append([])  # Add empty variation
                    variation_saved = True
                    
                elif key == ord('q'):  # Quit completely
                    if image_variations:  # Save what we have so far
                        scanpaths[filename] = {f"var{i+1}": var for i, var in enumerate(image_variations)}
                    save_scanpaths(scanpaths, output_dir)
                    cv2.destroyAllWindows()
                    return
            
            cv2.destroyAllWindows()
        
        # Store all variations for this image
        scanpaths[filename] = {
            f"var{i+1}": variation for i, variation in enumerate(image_variations)
        }
        
        print(f"Completed all variations for {filename}")
        print("-" * 50)
    
    # Save all scanpaths at the end
    save_scanpaths(scanpaths, output_dir)

def save_scanpaths(scanpaths, output_dir):
    output_path = os.path.join(output_dir, 'scanpaths.json')
    with open(output_path, 'w') as f:
        json.dump(scanpaths, f, indent=4)
    print(f"All scanpaths saved to {output_path}")

if __name__ == "__main__":
    image_dir = 'newWay/dataset/displayFonts/'
    output_dir = 'newWay/prompt/'  
    main(image_dir, output_dir)