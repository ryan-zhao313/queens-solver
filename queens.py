import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

img = cv2.imread("test/254.PNG")

# handle error is no file found
if img is None:
    raise ValueError("Image not found or unable to load.")

# Resize and preprocess the image
img = cv2.resize(img, (0,0), fx=0.65, fy=0.65)
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

def cluster_lines(lines, threshold=10):
    if not lines:
        return []

    lines.sort(key=lambda line: (line[1] + line[3]) / 2)
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
grid_size = len(clustered_horizontal) - 1

# Define the colors list
colors = [
    {"name": "purple", "hex": 0xbba3e2, "rgb": (187, 163, 226)},
    {"name": "orange", "hex": 0xffc992, "rgb": (255, 201, 146)},
    {"name": "blue", "hex": 0x96beff, "rgb": (150, 190, 255)},
    {"name": "light-green", "hex": 0xb3dfa0, "rgb": (179, 223, 160)},
    {"name": "light-grey", "hex": 0xdfdfdf, "rgb": (223, 223, 223)},
    {"name": "red", "hex": 0xff7b60, "rgb": (255, 123, 96)},
    {"name": "yellow", "hex": 0xe6f388, "rgb": (230, 243, 136)},
    {"name": "dark-grey", "hex": 0xb9b29e, "rgb": (185, 178, 158)},
    {"name": "light-pink", "hex": 0xdfa0bf, "rgb": (223, 160, 191)},
    {"name": "light-blue", "hex": 0xa3d2d8, "rgb": (163, 210, 216)},
    {"name": "black", "hex": 0x000000, "rgb": (0, 0, 0)}
]

# Function to convert a hex color to RGB
def hex_to_rgb(hex_value):
    return ((hex_value >> 16) & 0xFF, (hex_value >> 8) & 0xFF, hex_value & 0xFF)

# Function to calculate the closest color
def get_closest_color(rgb):
    r, g, b = map(int, rgb)
    min_distance = math.inf
    closest_color_name = 'black'

    for color in colors:
        cr, cg, cb = color["rgb"]
        distance = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = color["name"]
    return closest_color_name

# Map colors from the cropped image into a 2D grid
grid = np.full((grid_size, grid_size), 'black', dtype='<U15')

step_x = cropped_grid.shape[1] / grid_size
step_y = cropped_grid.shape[0] / grid_size

for row in range(grid_size):
    for col in range(grid_size):
        center_x = int((col + 0.5) * step_x)
        center_y = int((row + 0.5) * step_y)
        pixel_bgr = cropped_grid[center_y, center_x]
        pixel_rgb = (pixel_bgr[2], pixel_bgr[1], pixel_bgr[0])
        color_name = get_closest_color(pixel_rgb)
        grid[row, col] = color_name

def solve_queens(grid):
    # Partition the grid into regions by color
    region_coords = defaultdict(list)
    n = grid_size
    for y, row in enumerate(grid):
        for x, color in enumerate(row):
            region_coords[color].append((x, y))  # Group coordinates by color

    # Sort regions by size (smallest first)
    sorted_regions = sorted(region_coords.values(), key=len)

    def try_region(regions):
        if not regions:
            return []

        # Take the smallest region and remaining regions
        smallest, *remaining = regions

        # Try placing a queen in every cell of the smallest region
        for x, y in smallest:
            # Filter out invalid placements for remaining regions
            filtered = [
                [
                    (x2, y2) for x2, y2 in region
                    if x2 != x and y2 != y
                    and (abs(x2 - x) > 1 or abs(y2 - y) > 1)
                ]
                for region in remaining
            ]

            # If any region has no valid cells, skip this placement
            if any(len(region) == 0 for region in filtered):
                continue

            # Recurse on the remaining regions
            solution = try_region(filtered)
            if solution is not None:
                return [(x, y)] + solution
        return None

    # Generate all solutions
    return try_region(sorted_regions)

# Place the queens on a board
def display_solution(grid_size, solution):
    output_grid = [['-' for _ in range(grid_size)] for _ in range(grid_size)]
    for x, y in solution:
        output_grid[y][x] = '👑'
    for row in output_grid:
        print(' '.join(row))

solution = solve_queens(grid)
if solution:
    print("Solution found:")
    display_solution(grid_size, solution)
else:
    print("No solution exists.")