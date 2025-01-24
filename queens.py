import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread("test/254.PNG")

# handle error is no file found
if img is None:
    raise ValueError("Image not found or unable to load.")

# Resize and preprocess the image
img = cv2.resize(img, (0,0), fx=0.75, fy=0.75)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
edges = cv2.Canny(blurred_img, 50, 150)

# Find contours and crop the grid
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
grid_contour = max(contours, key=cv2.contourArea, default=None)

if grid_contour is None:
    raise ValueError("No grid contour found.")

x, y, w, h = cv2.boundingRect(grid_contour)
cropped_grid = img[y:y+h, x:x+w]

# Detect lines in the cropped grid
gray_cropped = cv2.cvtColor(cropped_grid, cv2.COLOR_BGR2GRAY)
edges_cropped = cv2.Canny(gray_cropped, 50, 150)

# # Original resized image
# plt.subplot(1, 2, 1)
# plt.title("Resized Image")
# plt.imshow(cv2.cvtColor(cropped_grid, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# # Edges detected
# plt.subplot(1, 2, 2)
# plt.title("Edges Detected")
# plt.imshow(edges_cropped, cmap="gray")
# plt.axis("off")
# plt.show()

def cluster_lines(lines, threshold=10):
    if not lines:
        return []

    lines.sort(key=lambda line: (line[1] + line[3]) / 2)  # Average y-coordinate
    clustered_lines = []
    cluster = [lines[0]]

    for i in range(1, len(lines)):
        prev_line = cluster[-1]
        curr_line = lines[i]
        distance = abs((curr_line[1] + curr_line[3]) / 2 - (prev_line[1] + prev_line[3]) / 2)

        if distance < threshold:
            cluster.append(curr_line)
        else:
            # Merge the cluster into one representative line
            clustered_lines.append(merge_lines(cluster))
            cluster = [curr_line]

    # Add the last cluster
    clustered_lines.append(merge_lines(cluster))
    return clustered_lines

def merge_lines(cluster):
    x1 = min(line[0] for line in cluster)
    x2 = max(line[2] for line in cluster)
    y1 = int(np.mean([line[1] for line in cluster]))
    y2 = int(np.mean([line[3] for line in cluster]))
    return (x1, y1, x2, y2)

# Perform Hough Line Transform to detect lines in the image
lines = cv2.HoughLinesP(
            edges_cropped,
            1,
            np.pi / 180,
            threshold=180,
            minLineLength=150,
            maxLineGap=20
            )

# lines = cv2.HoughLines(edges_cropped, 1, np.pi / 180, threshold=150)

# Create a copy of the resized image to draw the lines on
line_image = cropped_grid.copy()

# # Draw the lines
# for points in lines:
#     x1,y1,x2,y2=points[0]
#     cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)

horizontal_lines = []

if lines is not None:
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        theta = np.arctan2(y2 - y1, x2 - x1)
        if np.isclose(theta, 0, atol=np.pi/180):
            horizontal_lines.append(line)

# Cluster lines
clustered_horizontal = cluster_lines(horizontal_lines, threshold=15)

# Compute grid size
num_rows = len(clustered_horizontal) - 1
print(f"Grid size: {num_rows} rows x {num_rows} columns")

# Draw clustered lines on the image
for (x1, y1, x2, y2) in clustered_horizontal:
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Clustered Lines")
plt.show()