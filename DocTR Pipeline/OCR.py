import argparse
import json
import cv2
import matplotlib.pyplot as plt
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import numpy as np

# Function to process the OCR results and save the bounding boxes in JSON format
def process_ocr(image_path):
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]
    predictor = ocr_predictor(pretrained=True)

    # Read the image using DocumentFile from doctr
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)
    json_export = result.export()
    ocr_width, ocr_height = json_export['pages'][0]['dimensions']
    width_ratio = orig_width / ocr_width
    height_ratio = orig_height / ocr_height

    # Convert relative bounding box to absolute bounding box in xyxy format
    def relative_to_absolute_bbox(rel_bbox, width_ratio, height_ratio):
        x_min_rel, y_min_rel = rel_bbox[0]
        x_max_rel, y_max_rel = rel_bbox[1]
        x_min = int(x_min_rel * ocr_width * width_ratio)
        y_min = int(y_min_rel * ocr_height * height_ratio)
        x_max = int(x_max_rel * ocr_width * width_ratio)
        y_max = int(y_max_rel * ocr_height * height_ratio)
        return [x_min, y_min, x_max, y_max]

    # Collect words and their bounding boxes
    word_boxes = []
    for page in json_export['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                for word in line['words']:
                    abs_bbox = relative_to_absolute_bbox(word['geometry'], width_ratio, height_ratio)
                    word_boxes.append({"word": word['value'], "bounding_box": abs_bbox})

    # Desired JSON format
    output_json = word_boxes
    name = image_path[:image_path.index('.')]
    json_filename = f"{name}_text.json"
    with open(json_filename, "w") as f:
        json.dump(output_json, f, indent=4)

    print(f"JSON output saved to {json_filename}")


def inpaint_text(image_path, json_path, output_path):
    image = cv2.imread(image_path)
    with open(json_path, 'r') as f:
        bounding_boxes = json.load(f)

    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw white rectangles on the mask where the bounding boxes are located
    for item in bounding_boxes:
        bbox = item['bounding_box']
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (255), thickness=cv2.FILLED)

    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save the inpainted image
    cv2.imwrite(output_path, inpainted_image)


# Function to visualize the image with bounding boxes and texts
def visualize_bounding_boxes(image_path, json_path):
    image = cv2.imread(image_path)
    with open(json_path, 'r') as f:
        bounding_boxes = json.load(f)

    # Draw bounding boxes and texts on the image
    for item in bounding_boxes:
        word = item['word']
        bbox = item['bounding_box']
        x_min, y_min, x_max, y_max = bbox

        # Draw the bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, word, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Main function to handle command-line arguments and call the appropriate functions
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process and visualize OCR results.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--vis", action="store_true", help="Visualize the image with bounding boxes")
    args = parser.parse_args()

    # Process OCR and save JSON
    process_ocr(args.image_path)

    image_path = args.image_path
    name = image_path[:image_path.index('.')]
    json_filename = f"{name}_text.json"
    output_path = f"{name}_redacted.png"
    inpaint_text(args.image_path, json_filename, output_path)

    # If --vis flag is provided, visualize the bounding boxes
    if args.vis:
        visualize_bounding_boxes(args.image_path, json_filename)

# Entry point of the script
if __name__ == "__main__":
    main()
