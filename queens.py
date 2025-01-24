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

# Perform Hough Line Transform to detect lines in the image
lines = cv2.HoughLinesP(
            edges_cropped,
            1,
            np.pi / 180,
            threshold=150,
            minLineLength=50,
            maxLineGap=5
            )
# Create a copy of the resized image to draw the lines on
line_image = cropped_grid.copy()

# Draw the lines
lines_list = []
for points in lines:
    x1,y1,x2,y2=points[0]
    cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)

# Create a copy of the resized image to draw the lines on
# line_image = cropped_grid.copy()

plt.figure(figsize=(10, 10))
plt.title("Detected Grid Lines")
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Draw the detected lines on the image
# if lines is not None:
#     for rho, theta in lines[:, 0]:
#         a = np.cos(theta)
#         b = np.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

horizontal_lines = []
vertical_lines = []
if lines is not None:
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        theta = np.arctan2(y2 - y1, x2 - x1)
        if np.isclose(theta, 0, atol=np.pi/180):
            horizontal_lines.append(line)
        elif np.isclose(theta, np.pi / 2, atol=np.pi/180):
            vertical_lines.append(line)

# Apply clustering to horizontal and vertical lines separately
# def cluster_lines(lines):
#     lines_array = np.array(lines)
#     kmeans = KMeans(n_clusters=len(lines) // 2, random_state=42).fit(lines_array)
#     clustered_lines = kmeans.cluster_centers_
#     return clustered_lines

# clustered_horizontal = cluster_lines(horizontal_lines)
# clustered_vertical = cluster_lines(vertical_lines)

if lines is not None:
    for (x1, y1, x2, y2) in clustered_horizontal:
        cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for (x1, y1, x2, y2) in clustered_vertical:
        cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

# Display the image with the detected lines
# plt.figure(figsize=(10, 10))
# plt.title("Detected Grid Lines")
# plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()

# Convert the detected lines into a more usable form
if lines is not None:
    for rho, theta in lines[:, 0]:
        if np.isclose(theta, 0, atol=np.pi/180):
            horizontal_lines.append((rho, theta))
        elif np.isclose(theta, np.pi / 2, atol=np.pi/180):
            vertical_lines.append((rho, theta))

# Sort lines
horizontal_lines = sorted(horizontal_lines, key=lambda x: x[0])
vertical_lines = sorted(vertical_lines, key=lambda x: x[0])

# Find intersections
intersections = []
for h_rho, h_theta in horizontal_lines:
    for v_rho, v_theta in vertical_lines:
        # Calculate x, y using the line equations
        A = np.array([
            [np.cos(h_theta), np.sin(h_theta)],
            [np.cos(v_theta), np.sin(v_theta)]
        ])
        b = np.array([h_rho, v_rho])
        if np.linalg.det(A) != 0:  # Ensure the determinant is not zero
            x, y = np.linalg.solve(A, b)
            intersections.append((int(round(x)), int(round(y))))

# Convert intersections to a NumPy array for clustering
intersections = np.array(intersections)

# Apply clustering to group intersections
kmeans_intersections = KMeans(n_clusters=100, random_state=42).fit(intersections)
grouped_intersections = kmeans_intersections.cluster_centers_

# Round and convert to integer points
rounded_intersections = [tuple(map(int, point)) for point in grouped_intersections]

# Draw grid and intersections
grid_image = cropped_grid.copy()
for (x, y) in intersections:
    if 0 <= x < grid_image.shape[1] and 0 <= y < grid_image.shape[0]:  # Ensure points are within bounds
        cv2.circle(grid_image, (x, y), 5, (0, 255, 0), -1)

print(len(rounded_intersections))

plt.figure(figsize=(10, 10))
plt.title("Detected Grid Intersections")
plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Compute grid size
num_rows = len(horizontal_lines) - 1
num_cols = len(vertical_lines) - 1
print(f"Grid size: {num_rows} rows x {num_cols} columns")