import numpy as np
import matplotlib.pyplot as plt

def analyze_metrics(drones, model):
    """Analyze and plot metrics"""
    
    # 1. Heading disturbance
    headings = []
    n_steps = len(drones[0].path)
    
    for i in range(n_steps):
        heading_vec = np.array([0.0, 0.0])
        for j in range(model['n']):
            x = drones[j].path[i][2]  # heading
            heading_vec += np.array([np.cos(x), np.sin(x)])
        headings.append(np.linalg.norm(heading_vec) / model['n'])
    
    # Plot heading
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    steps = list(range(n_steps))
    ax1.scatter(steps, headings, s=10, alpha=0.7)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Order')
    ax1.set_xlim([0, n_steps])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Heading Order Parameter')
    
    # 2. Distance between drones
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    xmax = n_steps
    
    # Alert zone line
    ax2.axhline(y=drones[0].ra, color='k', linestyle='--', 
                linewidth=2, label='Alert Zone')
    
    # Plot distances between all pairs
    for i in range(model['n'] - 1):
        for j in range(i + 1, model['n']):
            dis = []
            for k in range(n_steps):
                pos_i = drones[i].path[k][:2]
                pos_j = drones[j].path[k][:2]
                dis.append(np.linalg.norm(pos_i - pos_j))
            
            ax2.plot(dis, linewidth=1.5, 
                    label=f'UAV{i+1}-UAV{j+1}')
    
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Distance [m]')
    ax2.set_xlim([0, xmax])
    ax2.legend(ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Inter-drone Distances')
    
    # 3. Mean distances (boxplot)
    dis_matrix = []
    labels = []
    
    for i in range(model['n'] - 1):
        dis = []
        for k in range(n_steps):
            pos_i = drones[i].path[k][:2]
            pos_j = drones[i+1].path[k][:2]
            dis.append(np.linalg.norm(pos_i - pos_j))
        
        dis_matrix.append(dis)
        labels.append(f'UAV{i+1}-UAV{i+2}')
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bp = ax3.boxplot(dis_matrix, widths=0.5, labels=labels, patch_artist=True)
    
    # Customize boxplot colors
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Distance [m]')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_title('Distance Distribution Between Consecutive UAVs')
    
    # 4. Error plot
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    # Calculate average error
    avg_errors = []
    for k in range(n_steps):
        errors = []
        for i in range(model['n'] - 1):
            pos_i = drones[i].path[k][:2]
            pos_j = drones[i+1].path[k][:2]
            errors.append(np.linalg.norm(pos_i - pos_j))
        
        avg_error = 0.5 * (np.mean(errors) - model['d'])
        avg_errors.append(avg_error)
    
    ax4.plot(avg_errors, linewidth=2)
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Average error [m]')
    ax4.set_xlim([0, xmax])
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Formation Tracking Error')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'headings': headings,
        'dis_matrix': dis_matrix,
        'avg_errors': avg_errors
    }

if __name__ == "__main__":
    # This would be called after main simulation
    # For testing, you would need to run main() first
    print("Run main.py first, then call analyze_metrics() with the drones and model")