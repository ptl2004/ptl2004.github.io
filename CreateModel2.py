import numpy as np

def CreateModel2():
    """CREATE MODEL - Exact MATLAB translation"""
    # Formation description
    d = 0.8        # The desired distance
    alpha = 3 * np.pi / 4  # The desired angle
    n = 5          # Number of drones

    # Start and Goal
    start = np.array([-20.0, 3.5])
    goal = np.array([22.0, 3.5])

    # Obstacles set
    obs1 = np.array([[-10.0, 0.0], 
                     [-10.0, 1.0], 
                     [0.0, 3.15], 
                     [15.0, 3.15], 
                     [20.0, 0.0]])
    
    obs2 = np.array([[-10.0, 7.0], 
                     [-10.0, 6.0], 
                     [10.0, 3.85], 
                     [15.0, 3.85], 
                     [20.0, 7.0]])

    # Limit
    xmax = 22.0
    xmin = -24.0
    ymax = 7.0
    ymin = 0.0

    # Scenario
    model = {
        'd': d,
        'alpha': alpha,
        'n': n,
        'start': start,
        'goal': goal,
        'obstacles': [obs1, obs2],
        'xmax': xmax,
        'xmin': xmin,
        'ymax': ymax,
        'ymin': ymin
    }
    
    return model

if __name__ == "__main__":
    model = CreateModel2()
    print("Model created successfully!")