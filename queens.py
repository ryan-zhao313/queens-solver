import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread("test/254.PNG")

# handle error is no file found
if img is None:
    raise ValueError("Image not found or unable to load.")

# Resize and preprocess the image
img = cv2.resize(img, (600, 600))
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

# Original resized image
# plt.subplot(1, 2, 1)
# plt.title("Resized Image")
# plt.imshow(cv2.cvtColor(cropped_grid, cv2.COLOR_BGR2RGB))
# plt.axis("off")

# Edges detected
# plt.subplot(1, 2, 2)
# plt.title("Edges Detected")
# plt.imshow(edges_cropped, cmap="gray")
# plt.axis("off")

# Perform Hough Line Transform to detect lines in the image
lines = cv2.HoughLines(edges_cropped, 1, np.pi / 180, threshold= 200)

# Create a copy of the resized image to draw the lines on
line_image = cropped_grid.copy()

# Draw the detected lines on the image
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image with the detected lines
plt.figure(figsize=(10, 10))
plt.title("Detected Grid Lines")
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
# plt.show()

# Convert the detected lines into a more usable form
horizontal_lines = []
vertical_lines = []

# if lines is not None:
#     for rho, theta in lines[:, 0]:
#         if np.isclose(theta, 0):  # Horizontal lines (theta ~ 0)
#             horizontal_lines.append(rho)
#         elif np.isclose(theta, np.pi / 2):  # Vertical lines (theta ~ pi/2)
#             vertical_lines.append(rho)

if lines is not None:
    for rho, theta in lines[:, 0]:
        if np.isclose(theta, 0, atol=np.pi / 180 * 10):  # Horizontal
            horizontal_lines.append(rho)
        elif np.isclose(theta, np.pi / 2, atol=np.pi / 180 * 10):  # Vertical
            vertical_lines.append(rho)

# Cluster the lines to remove duplicates
def cluster_lines(lines, num_clusters):
    lines = np.array(lines).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(lines)
    clustered_lines = sorted(kmeans.cluster_centers_.flatten())
    return clustered_lines

# Cluster lines into 10 (9 grid lines + 1 boundary line)
horizontal_lines = cluster_lines(horizontal_lines, 10)
vertical_lines = cluster_lines(vertical_lines, 10)

# Find intersections
intersections = []
for h in horizontal_lines:
    for v in vertical_lines:
        x = int(v)
        y = int(h)
        intersections.append((x, y))

# Draw grid and intersections
grid_image = cropped_grid.copy()
for (x, y) in intersections:
    cv2.circle(grid_image, (x, y), 5, (0, 255, 0), -1)

plt.figure(figsize=(10, 10))
plt.title("Grid")
plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Compute grid size
num_rows = len(horizontal_lines) - 1
num_cols = len(vertical_lines) - 1
print(f"Grid size: {num_rows} rows x {num_cols} columns")