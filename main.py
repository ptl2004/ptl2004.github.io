"""
MAIN.PY - Exact equivalent of MATLAB main.m
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # For proper display
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from CreateModel2 import CreateModel2
from Drone import Drone

def main():
    """EXACT MATLAB main.m algorithm"""
    # Load model
    model = CreateModel2()
    
    # Init drones - EXACT MATLAB: model.start+unifrnd(-1,1,[1,2])
    drones = []
    print("Initializing drones...")
    for i in range(1, model['n'] + 1):
        random_offset = np.random.uniform(-1, 1, size=(2,))
        start_pos = np.array(model['start']) + random_offset
        drone = Drone(i, start_pos)
        drones.append(drone)
        print(f"  UAV{i} at: [{start_pos[0]:.2f}, {start_pos[1]:.2f}]")
    
    dt = 0.02
    iter_count = 0
    reached = False
    
    print("\nStarting simulation...")
    # MATLAB: while ~reached
    while not reached:
        reached_sum = 0
        
        # MATLAB: for i = 1:model.n
        for i in range(model['n']):
            vel, reach = drones[i].Behavior(drones, model)
            # MATLAB: reached = reached + reach;
            reached_sum += int(reach)
            drones[i].UpdatePosition(vel, dt)
        
        iter_count += 1
        
        # Progress display
        if iter_count % 100 == 0:
            leader_idx = (model['n'] + 1) // 2 - 1
            leader_pos = drones[leader_idx].position
            goal_dist = np.linalg.norm(np.array(model['goal']) - leader_pos)
            print(f"Iteration {iter_count}: Distance to goal = {goal_dist:.2f}m")
        
        # MATLAB: reached = reached + reach (sum of boolean values)
        reached = reached_sum > 0
    
    print(f"\nSimulation completed in {iter_count} iterations")
    
    # Create figure - EXACT MATLAB plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    # Start location - MATLAB: plot(xs,ys,'bs','MarkerSize',10,'MarkerFaceColor','b')
    xs, ys = model['start']
    ax.plot(xs, ys, 'bs', markersize=10, markerfacecolor='b', 
            markeredgecolor='darkblue', markeredgewidth=2, label='Start')
    
    # End location - MATLAB: plot(xf,yf,'rp','MarkerSize',10,'MarkerFaceColor','r')
    xf, yf = model['goal']
    ax.plot(xf, yf, 'rp', markersize=10, markerfacecolor='r',
            markeredgecolor='darkred', markeredgewidth=2, label='Goal')
    
    # Obstacles - MATLAB: pgon = polyshape(obs(:,1), obs(:,2)); plot(pgon, 'FaceColor','black');
    for obs in model['obstacles']:
        polygon = Polygon(obs, closed=True, facecolor='black', alpha=0.7)
        ax.add_patch(polygon)
    
    # Plot path - MATLAB: plot(drones(i).path(:,1), drones(i).path(:,2), 'LineWidth', 2);
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(model['n']):
        path_array = np.array(drones[i].path)
        ax.plot(path_array[:, 0], path_array[:, 1], 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'UAV{i+1}')
    
    # Plot formation at specific steps - EXACT MATLAB algorithm
    num = 6
    if len(drones[0].path) > 0:
        step = max(1, len(drones[0].path) // num)
        
        for i in range(num + 1):
            idx = i * step
            if idx >= len(drones[0].path):
                idx = len(drones[0].path) - 1
            
            # Plot formation graph - MATLAB: gr = [gr; drones(j).path(idx,:)];
            gr = []
            for j in range(model['n']):
                gr.append(drones[j].path[idx])
            
            gr_array = np.array(gr)
            # MATLAB: plot(gr(:,1), gr(:,2), '-k', 'LineWidth', 2);
            ax.plot(gr_array[:, 0], gr_array[:, 1], 
                   '-k', linewidth=2, alpha=0.5)
            
            # Plot formation agent
            for j in range(model['n']):
                # MATLAB scatter parameters
                ax.scatter(drones[j].path[idx][0], drones[j].path[idx][1],
                          marker='o',
                          edgecolor=[0, 0.5, 0.5],
                          facecolor=[0, 0.7, 0.7],
                          linewidth=1.5,
                          s=80)  # markersize equivalent
    
    # MATLAB: xlabel('x [m]'); ylabel('y [m]'); axis('equal');
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_aspect('equal')
    
    # MATLAB: xlim([model.xmin model.xmax]); ylim([model.ymin model.ymax]);
    ax.set_xlim([model['xmin'], model['xmax']])
    ax.set_ylim([model['ymin'], model['ymax']])
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('V-Shape Formation Navigation', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('formation_result.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'formation_result.png'")
    
    # Show plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total iterations: {iter_count}")
    print(f"Total time: {iter_count * dt:.2f} seconds")
    
    for i, drone in enumerate(drones):
        path_array = np.array(drone.path)
        distance = np.sum(np.linalg.norm(np.diff(path_array[:, :2], axis=0), axis=1))
        print(f"UAV{i+1}: {len(path_array)} points, {distance:.2f}m traveled")
    
    return drones, model

if __name__ == "__main__":
    try:
        drones, model = main()
        print("\n✓ Simulation completed successfully!")
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()