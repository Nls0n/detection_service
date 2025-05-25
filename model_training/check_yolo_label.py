import cv2
import numpy as np


def visualize_yolo_labels(image_path, label_path, output_path=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    h, w = image.shape[:2]

    class_names = ["0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9",
                   "10", "11", "12"]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
              (0, 255, 255), (255, 0, 255), (128, 128, 0), (0, 128, 128),
              (128, 0, 128), (64, 64, 64), (192, 192, 192), (64, 0, 0),
              (0, 64, 0)]

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Label file not found {label_path}")
        return

    if not lines:
        print(f"Empty label file: {label_path}")
        return

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:5])

        # Коэффициент уменьшения (0.2 = 20% от исходного размера)
        scale_factor = 0.2 if class_id not in [6, 7, 8] else 1.0

        abs_x_center = x_center * w
        abs_y_center = y_center * h
        abs_width = width * w * scale_factor
        abs_height = height * h * scale_factor

        x1 = max(0, int(abs_x_center - abs_width / 2))
        y1 = max(0, int(abs_y_center - abs_height / 2))
        x2 = min(w, int(abs_x_center + abs_width / 2))
        y2 = min(h, int(abs_y_center + abs_height / 2))

        color = colors[class_id % len(colors)]
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{class_name} ({class_id})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if len(parts) > 5:
            points = np.array([float(x) for x in parts[5:]], dtype=np.float32).reshape(-1, 2)
            points *= np.array([w, h])

            if class_id not in [6, 7, 8]:
                centroid = np.mean(points, axis=0)
                points = centroid + (points - centroid) * scale_factor

            cv2.polylines(image, [points.astype(int)], isClosed=True, color=color, thickness=2)

    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow('YOLO Visualization', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


tile_image_path = "datasetx/images/train/0-640-ls-14-d01_18.jpg"
tile_label_path = 'datasetx/labels/train/0-640-ls-14-d01_18.txt'
visualize_yolo_labels(tile_image_path, tile_label_path, 'visualization.jpg')