# Queens LinkedIn Game Solver
This solves the LinkedIn Queens Game in Python. It uses OpenCV to detect the game grid and then solves it using optimized backtracking.

## Getting Started
Clone the repo and install all the Python libraries
```pip install -r requirements.txt```

## How It Works
- Take a screenshot of the LInkedIn Queens board
- Image is preprocessed
- Hough Lines Transform is preformed to detect the lines in the grid
- Cluster and merge the lines to find the size of the game board
- Map all the colors in the grid
- Partition the grid by color regions
- Backtracking algorithm to place the Queens on each available cell
- Output first solution found

## Testing Code
![Edge Detection](/img/edge-detection.PNG)
```
# Original resized image
plt.subplot(1, 2, 1)
plt.title("Resized Image")
plt.imshow(cv2.cvtColor(cropped_grid, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Edges detected
plt.subplot(1, 2, 2)
plt.title("Edges Detected")
plt.imshow(edges_cropped, cmap="gray")
plt.axis("off")
plt.show()
```

![CLustered Lines](/img/clustered-lines.PNG)
![Hough Lines](/img/detected-grid-lines-corrected.PNG)
```
lines = cv2.HoughLines(edges_cropped, 1, np.pi / 180, threshold=150)

# Create a copy of the resized image to draw the lines on
line_image = cropped_grid.copy()

# Draw the lines
for points in lines:
    x1,y1,x2,y2=points[0]
    cv2.line(line_image,(x1,y1),(x2,y2),(0,255,0),2)

# Draw clustered lines on the image
for (x1, y1, x2, y2) in clustered_horizontal:
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Clustered Lines")
plt.show()

print(f"Grid size: {grid_size} rows x {grid_size} columns")
```

