import numpy as np

def translate_object(points, tx, ty):
    # Handle empty input
    if not points:
        return []
    
    # Define the 3x3 2D translation matrix
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=float)
    
    # Convert points to homogeneous coordinates by appending a column of 1s
    points_arr = np.array(points, dtype=float)
    ones = np.ones((points_arr.shape[0], 1))
    homogeneous_points = np.hstack([points_arr, ones])
    
    # Apply the translation matrix (transpose to multiply 3x3 by 3xN, then transpose back)
    translated_homogeneous = (translation_matrix @ homogeneous_points.T).T
    
    # Drop the homogeneous coordinate (the 3rd column) and convert back to a standard Python list
    translated_points = translated_homogeneous[:, :2].tolist()
    
    return translated_points