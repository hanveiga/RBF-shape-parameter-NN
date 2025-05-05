import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import random as rand
from scipy.spatial.distance import cdist

def phi(f, x, y):
    z = (1 + (f * np.linalg.norm(x-y)) ** 2) ** (-0.5)
    return z

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def find_nearest_neighbors(points):
    distance_matrix = cdist(points, points)
    np.fill_diagonal(distance_matrix, np.inf)
    return np.min(distance_matrix, axis=1)
    
# Function to compute the Euclidean distance between two points
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Function to check if a point is inside a circle
def is_inside_circle(p, c):
    return dist(p, c[:2]) <= c[2]

# Function to compute the circle from 3 points
def circle_from_3_points(p1, p2, p3):
    A = p2[0] - p1[0]
    B = p2[1] - p1[1]
    C = p3[0] - p1[0]
    D = p3[1] - p1[1]
    E = A * (p1[0] + p2[0]) + B * (p1[1] + p2[1])
    F = C * (p1[0] + p3[0]) + D * (p1[1] + p3[1])
    G = 2 * (A * (p3[1] - p2[1]) - B * (p3[0] - p2[0]))
    
    if G == 0:  # Collinear points
        return None
    
    cx = (D * E - B * F) / G
    cy = (A * F - C * E) / G
    r = dist((cx, cy), p1)
    
    return (cx, cy, r)

# Function to compute the circle from 2 points
def circle_from_2_points(p1, p2):
    cx = (p1[0] + p2[0]) / 2
    cy = (p1[1] + p2[1]) / 2
    r = dist(p1, p2) / 2
    return (cx, cy, r)

# Recursive function to find the minimum enclosing circle
def welzl(P, R, n):
    if n == 0 or len(R) == 3:
        if len(R) == 0:
            return (0, 0, 0)
        elif len(R) == 1:
            return (R[0][0], R[0][1], 0)
        elif len(R) == 2:
            return circle_from_2_points(R[0], R[1])
        else:
            return circle_from_3_points(R[0], R[1], R[2])
    
    idx = rand.randint(0, n - 1)
    p = P[idx]
    P[idx], P[n - 1] = P[n - 1], P[idx]
    
    d = welzl(P, R, n - 1)
    
    if is_inside_circle(p, d):
        return d
    
    return welzl(P, R + [p], n - 1)

# Function to find the minimum enclosing circle
def find_min_circle(points):
    P = points[:]
    #random.shuffle(P)
    return welzl(P, [], len(P))

# Create a synthetic checkerboard image
def create_checkerboard(size=(200, 200), num_checks=10):
    # Create a checkerboard pattern
    x = np.linspace(0, num_checks, size[1])
    y = np.linspace(0, num_checks, size[0])
    X, Y = np.meshgrid(x, y)
    checkerboard = ((np.floor(X) + np.floor(Y)) % 2)
    return checkerboard

def load_image(path,size=(200, 200)):
    #image_array = plt.imread(path)
    image = Image.open(path).convert('L')
    resized_image = image.resize(size)
    image_array = np.array(resized_image)/ 255.0
    return image_array

def display_images(image,file_name):

    fig = plt.figure(figsize=(10, 6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(f'{file_name}',bbox_inches='tight', dpi=150)
    plt.close(fig)

def load_image(path):
    #image_array = plt.imread(path)
    image = Image.open(path).convert('L')
    resized_image = image.resize((200, 200))
    image_array = np.array(resized_image)/ 255.0
    return image_array
