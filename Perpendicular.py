import numpy as np

def Perpendicular(x, a, b):
    """PERPENDICULAR - Exact MATLAB translation"""
    d_ab = np.linalg.norm(a - b)
    d_ax = np.linalg.norm(a - x)
    d_bx = np.linalg.norm(b - x)

    if d_ab != 0:
        # Check if x is between a and b
        if np.dot(a - b, x - b) * np.dot(b - a, x - a) >= 0:
            # x is between a and b
            px = b[0] - a[0]
            py = b[1] - a[1]
            dAB = px * px + py * py
            
            u = ((x[0] - a[0]) * px + (x[1] - a[1]) * py) / dAB
            p = np.array([a[0] + u * px, a[1] + u * py])
        else:
            # x is not between a and b
            if d_ax < d_bx:
                p = np.array(a)
            else:
                p = np.array(b)
    else:
        # a and b are identical
        p = np.array(a)
    
    return p

if __name__ == "__main__":
    # Test
    x = np.array([0, 0])
    a = np.array([-1, 1])
    b = np.array([1, 1])
    result = Perpendicular(x, a, b)
    print(f"Perpendicular point: {result}")