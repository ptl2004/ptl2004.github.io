"""
METRICS_ANALYSIS.PY - Fixed version with Unicode and matplotlib fixes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from CreateModel2 import CreateModel2
from Drone import Drone
import os
import sys

def run_simulation_with_metrics(model, max_frames=4000):
    """Run simulation and collect all data for video"""
    print("Running simulation with metrics collection...")
    
    # Initialize drones
    drones = []
    for i in range(1, model['n'] + 1):
        random_offset = np.random.uniform(-1, 1, size=(2,))
        start_pos = np.array(model['start']) + random_offset
        drone = Drone(i, start_pos)
        drones.append(drone)
    
    dt = 0.02
    reached = False
    
    # Store all data for analysis
    all_positions = [[] for _ in drones]
    all_velocities = [[] for _ in drones]
    all_heading_angles = [[] for _ in drones]
    
    frame_metrics = {
        'pairwise_distances': [],
        'formation_error': [],
        'consecutive_distances': [],
        'order_metric': [],
        'reconfiguration_count': [],
        'collision_avoidance_count': [],
        'time': []
    }
    
    frame = 0
    while not reached and frame < max_frames:
        reached_sum = 0
        reconfig_count = 0
        collision_avoid_count = 0
        
        for i in range(model['n']):
            # FIXED: Behavior now returns 3 values
            vel, reach, behaviors = drones[i].Behavior(drones, model)
            drones[i].UpdatePosition(vel, dt)
            reached_sum += int(reach)
            
            # Store position and velocity
            all_positions[i].append(drones[i].position.copy())
            all_velocities[i].append(vel.copy())
            
            # Calculate heading angle from velocity
            if np.linalg.norm(vel) > 0.001:
                heading = np.arctan2(vel[1], vel[0])
            else:
                heading = 0.0
            all_heading_angles[i].append(heading)
            
            # Count UAVs activating behaviors
            if behaviors.get('reconfig_active', False):
                reconfig_count += 1
            if behaviors.get('collision_avoid_active', False):
                collision_avoid_count += 1
        
        # Calculate metrics for current frame
        if frame < len(all_positions[0]):
            current_positions = [pos_list[frame] for pos_list in all_positions]
            current_headings = [heading_list[frame] for heading_list in all_heading_angles]
            
            # 1. Pairwise distances between UAVs
            pairwise_dists = []
            for i in range(model['n']):
                for j in range(i+1, model['n']):
                    dist = np.linalg.norm(current_positions[i] - current_positions[j])
                    pairwise_dists.append(dist)
            frame_metrics['pairwise_distances'].append(pairwise_dists)
            
            # 2. Average formation error
            formation_error = calculate_formation_error(current_positions, model, drones[0])
            frame_metrics['formation_error'].append(formation_error)
            
            # 3. Distances between consecutive UAVs
            cons_dists = []
            for i in range(model['n'] - 1):
                dist = np.linalg.norm(current_positions[i] - current_positions[i+1])
                cons_dists.append(dist)
            frame_metrics['consecutive_distances'].append(np.mean(cons_dists) if cons_dists else 0)
            
            # 4. Order metric Φ
            order_metric = calculate_order_metric(current_headings)
            frame_metrics['order_metric'].append(order_metric)
            
            # 5. Number of UAVs reconfiguring
            frame_metrics['reconfiguration_count'].append(reconfig_count)
            
            # 6. Number of UAVs avoiding collisions
            frame_metrics['collision_avoidance_count'].append(collision_avoid_count)
            
            # 7. Time
            frame_metrics['time'].append(frame * dt)
        
        reached = reached_sum > 0
        frame += 1
        
        if frame % 500 == 0:
            print(f"  Frame {frame}")
    
    total_frames = frame
    print(f"Simulation completed: {total_frames} frames")
    
    return all_positions, frame_metrics, total_frames, drones

def calculate_formation_error(positions, model, leader_drone):
    """Calculate formation error from desired positions"""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Leader position (middle UAV)
    leader_idx = n // 2
    leader_pos = positions[leader_idx]
    
    # Calculate desired positions for each UAV
    desired_positions = []
    for i in range(n):
        if i == leader_idx:
            desired_positions.append(leader_pos)
        else:
            # Desired distance and angle
            d_i = model.get('d', 1.0) * abs(leader_idx - i)
            alpha = model.get('alpha', 3*np.pi/4)
            
            # Determine wing
            if i < leader_idx:
                wing_angle = alpha  # Left wing
            else:
                wing_angle = -alpha  # Right wing
            
            # Leader heading
            leader_heading = leader_drone.heading_angle
            
            desired_pos = leader_pos + d_i * np.array([
                np.cos(leader_heading + wing_angle),
                np.sin(leader_heading + wing_angle)
            ])
            desired_positions.append(desired_pos)
    
    # Calculate average error (excluding leader)
    errors = []
    for i in range(n):
        if i != leader_idx:
            error = np.linalg.norm(positions[i] - desired_positions[i])
            errors.append(error)
    
    return np.mean(errors) if errors else 0.0

def calculate_order_metric(headings):
    """Calculate order metric Φ"""
    n = len(headings)
    if n == 0:
        return 0.0
    
    # Sum of unit vectors
    sum_vector = np.array([0.0, 0.0])
    for angle in headings:
        sum_vector += np.array([np.cos(angle), np.sin(angle)])
    
    # Normalize
    phi = np.linalg.norm(sum_vector) / n
    
    return phi

def create_evaluation_plots(metrics, total_frames, model, save_path='evaluation_plots'):
    """Create 4 evaluation plots like in the paper"""
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    # Time (seconds)
    time = metrics['time'][:total_frames]
    
    # 1. Figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluation of the Proposed Algorithm', fontsize=16, fontweight='bold')
    
    # (a) Distances between each pair of UAVs over time
    ax1 = axes[0, 0]
    pairwise_dists = metrics['pairwise_distances'][:total_frames]
    
    if pairwise_dists and len(pairwise_dists[0]) > 0:
        # Plot all pairs (limit to 10 for clarity)
        for pair_idx in range(min(10, len(pairwise_dists[0]))):
            distances = [dist_list[pair_idx] for dist_list in pairwise_dists 
                        if pair_idx < len(dist_list)]
            ax1.plot(time[:len(distances)], distances, linewidth=1, alpha=0.5)
    
    # Alert radius and sensing radius lines
    ax1.axhline(y=0.3, color='r', linestyle='--', linewidth=2, label='Alert radius (0.3m)')
    ax1.axhline(y=2.0, color='orange', linestyle=':', linewidth=1.5, label='Sensing radius (2.0m)')
    
    # Desired distance line
    desired_d = model.get('d', 1.0)
    ax1.axhline(y=desired_d, color='g', linestyle='-', linewidth=1.5, alpha=0.5, 
                label=f'Desired distance ({desired_d}m)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('(a) Distances between each pair of UAVs over time')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim([0, 3])
    
    # (b) The average distance error of UAV formation
    ax2 = axes[0, 1]
    formation_error = metrics['formation_error'][:total_frames]
    if len(formation_error) > 0:
        ax2.plot(time[:len(formation_error)], formation_error, 'b-', linewidth=2)
        ax2.fill_between(time[:len(formation_error)], 0, formation_error, alpha=0.3)
    
    ax2.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Formation error (m)')
    ax2.set_title('(b) The average distance error of UAV formation')
    ax2.grid(True, alpha=0.3)
    if formation_error:
        ax2.set_ylim([0, max(formation_error)*1.1])
    
    # (c) The distance between consecutive UAVs
    ax3 = axes[1, 0]
    cons_distances = metrics['consecutive_distances'][:total_frames]
    if len(cons_distances) > 0:
        ax3.plot(time[:len(cons_distances)], cons_distances, 'g-', linewidth=2)
    
    # Desired distance line
    ax3.axhline(y=desired_d, color='r', linestyle='--', linewidth=2, 
                label=f'Desired distance ({desired_d}m)')
    
    # Tolerance band
    ax3.axhspan(desired_d*0.9, desired_d*1.1, alpha=0.1, color='green', 
                label='±10% tolerance')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance (m)')
    ax3.set_title('(c) The average distance between consecutive UAVs')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([0, desired_d*1.5])
    
    # (d) Values of the order metric Φ over time
    ax4 = axes[1, 1]
    order_metric = metrics['order_metric'][:total_frames]
    if len(order_metric) > 0:
        ax4.plot(time[:len(order_metric)], order_metric, 'm-', linewidth=2)
    
    # Order metric thresholds
    ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Perfect order (Φ=1)')
    ax4.axhline(y=0.9, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Good order (Φ>0.9)')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Order metric Φ')
    ax4.set_title('(d) Values of the order metric Φ over time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure_6_evaluation.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/figure_6_evaluation.pdf', bbox_inches='tight')
    plt.close(fig)  # Close figure instead of showing
    
    # 2. Behavior activation plots
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # (a) Reconfiguration activation
    reconfig_count = metrics['reconfiguration_count'][:total_frames]
    if len(reconfig_count) > 0:
        ax2_1.plot(time[:len(reconfig_count)], reconfig_count, 'r-', linewidth=2)
        ax2_1.fill_between(time[:len(reconfig_count)], 0, reconfig_count, 
                          alpha=0.3, color='red')
    
    ax2_1.set_xlabel('Time (s)')
    ax2_1.set_ylabel('Number of UAVs')
    ax2_1.set_title('(a) Number of UAVs activating reconfiguration behaviors over time')
    ax2_1.grid(True, alpha=0.3)
    ax2_1.set_ylim([0, model['n'] + 0.5])
    
    # (b) Collision avoidance activation
    collision_count = metrics['collision_avoidance_count'][:total_frames]
    if len(collision_count) > 0:
        ax2_2.plot(time[:len(collision_count)], collision_count, 'b-', linewidth=2)
        ax2_2.fill_between(time[:len(collision_count)], 0, collision_count, 
                          alpha=0.3, color='blue')
    
    ax2_2.set_xlabel('Time (s)')
    ax2_2.set_ylabel('Number of UAVs')
    ax2_2.set_title('(b) Number of UAVs activating collision avoidance behaviors over time')
    ax2_2.grid(True, alpha=0.3)
    ax2_2.set_ylim([0, model['n'] + 0.5])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/behavior_activation.png', dpi=300)
    plt.close(fig2)
    
    # 3. Histogram of distances
    fig3, ax3_1 = plt.subplots(figsize=(10, 6))
    
    # Collect all distances
    all_distances = []
    for dist_list in metrics['pairwise_distances'][:total_frames]:
        all_distances.extend(dist_list)
    
    if all_distances:
        ax3_1.hist(all_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3_1.axvline(x=0.3, color='r', linestyle='--', linewidth=2, label='Alert radius (0.3m)')
        ax3_1.axvline(x=desired_d, color='g', linestyle='--', linewidth=2, 
                     label=f'Desired distance ({desired_d}m)')
        
        # Statistics
        mean_dist = np.mean(all_distances)
        ax3_1.axvline(x=mean_dist, color='orange', linestyle='-', linewidth=2, 
                     label=f'Mean: {mean_dist:.2f}m')
        
        ax3_1.set_xlabel('Distance between UAVs (m)')
        ax3_1.set_ylabel('Frequency')
        ax3_1.set_title(f'Histogram of distances between UAV pairs')
        ax3_1.grid(True, alpha=0.3)
        ax3_1.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/distance_histogram.png', dpi=300)
    plt.close(fig3)
    
    return True

def create_statistics_table(metrics, model, drones, save_path):
    """Create statistical table - FIXED Unicode issue"""
    
    # Get metrics
    formation_errors = metrics['formation_error']
    pairwise_dists_all = metrics['pairwise_distances']
    cons_dists = metrics['consecutive_distances']
    order_metrics = metrics['order_metric']
    
    # Calculate statistics from last 100 frames (steady state)
    if len(formation_errors) > 100:
        steady_state_start = len(formation_errors) - 100
        avg_error = np.mean(formation_errors[steady_state_start:])
        avg_cons_distance = np.mean(cons_dists[steady_state_start:])
        avg_order_metric = np.mean(order_metrics[steady_state_start:])
    else:
        avg_error = np.mean(formation_errors) if formation_errors else 0
        avg_cons_distance = np.mean(cons_dists) if cons_dists else 0
        avg_order_metric = np.mean(order_metrics) if order_metrics else 0
    
    # Find minimum distance between UAVs
    min_distances = []
    for dist_list in pairwise_dists_all:
        if dist_list:
            min_distances.append(min(dist_list))
    min_distance = np.min(min_distances) if min_distances else 0
    
    # FIXED: Use English instead of Greek characters for file writing
    with open(f'{save_path}/results_summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SIMULATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Simulation Parameters:\n")
        f.write(f"  Number of UAVs: {model['n']}\n")
        f.write(f"  Desired distance d: {model.get('d', 1.0):.3f}m\n")
        f.write(f"  Desired angle alpha: {model.get('alpha', 3*np.pi/4)/np.pi:.3f}π rad\n")
        f.write(f"  Simulation duration: {metrics['time'][-1]:.1f}s ({len(metrics['time'])} frames)\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Average formation error: {avg_error:.5f}m\n")
        f.write(f"  Minimum inter-UAV distance: {min_distance:.5f}m\n")
        f.write(f"  Average consecutive UAV distance: {avg_cons_distance:.5f}m\n")
        f.write(f"  Average order metric Phi: {avg_order_metric:.3f}\n\n")
        
        f.write("Safety Assessment:\n")
        f.write(f"  Collision avoidance: {'SAFE' if min_distance > 0.3 else 'UNSAFE'}\n")
        f.write(f"  Formation stability: {'STABLE' if avg_error < 0.15 else 'UNSTABLE'}\n")
        f.write(f"  UAV alignment: {'GOOD' if avg_order_metric > 0.9 else 'POOR'}\n")
    
    # Print to console with safe encoding
    print("\n" + "="*100)
    print("STATISTICAL EVALUATION RESULTS")
    print("="*100)
    print(f"{'Metric':<30} {'Value':<15} {'Unit':<10} {'Target':<15} {'Status':<10}")
    print("-"*100)
    print(f"{'Number of UAVs':<30} {model['n']:<15} {'UAVs':<10} {'-':<15} {'OK':<10}")
    print(f"{'Desired distance (d)':<30} {model.get('d', 1.0):<15.3f} {'m':<10} {model.get('d', 1.0):<15.1f} {'Set':<10}")
    # Use 'alpha' instead of Greek character
    alpha_val = model.get('alpha', 3*np.pi/4)/np.pi
    print(f"{'Desired angle (alpha)':<30} {alpha_val:<15.3f}π {'rad':<10} {3/4:<15.2f}π {'Set':<10}")
    print(f"{'Average formation error':<30} {avg_error:<15.5f} {'m':<10} {'< 0.15':<15} {'✓' if avg_error < 0.15 else '⚠':<10}")
    print(f"{'Min distance between UAVs':<30} {min_distance:<15.5f} {'m':<10} {'> 0.3':<15} {'✓' if min_distance > 0.3 else '✗':<10}")
    print(f"{'Avg consecutive distance':<30} {avg_cons_distance:<15.5f} {'m':<10} {model.get('d', 1.0):<15.1f} {'✓' if abs(avg_cons_distance - model.get('d', 1.0)) < 0.1 else '⚠':<10}")
    print(f"{'Order metric (Phi)':<30} {avg_order_metric:<15.3f} {'-':<10} {'~1.0':<15} {'✓' if avg_order_metric > 0.9 else '⚠':<10}")
    print("="*100)

def main():
    """Main function to run evaluation"""
    print("="*80)
    print("DRONE FORMATION METRICS EVALUATION SYSTEM")
    print("(Generating plots similar to Figure 6 in the paper)")
    print("="*80)
    
    # Create model
    model = CreateModel2()
    
    # Add V-shape parameters if not present
    if 'd' not in model:
        model['d'] = 0.8
    if 'alpha' not in model:
        model['alpha'] = 3*np.pi/4
    
    # Use safe encoding for console output
    alpha_display = model.get('alpha', 3*np.pi/4)/np.pi
    print(f"\nModel parameters:")
    print(f"   • Number of UAVs: {model['n']}")
    print(f"   • Desired distance d: {model['d']}m")
    print(f"   • Desired angle alpha: {alpha_display:.2f}π rad")
    print(f"   • Start position: {model['start']}")
    print(f"   • Goal position: {model['goal']}")
    
    print(f"\nRunning simulation with metrics collection...")
    
    # Run simulation and collect metrics
    positions, metrics, total_frames, drones = run_simulation_with_metrics(model, max_frames=4000)
    
    print(f"\nSimulation completed successfully!")
    print(f"   • Total frames: {total_frames}")
    print(f"   • Total time: {total_frames*0.02:.1f}s")
    
    print(f"\nGenerating evaluation plots...")
    
    # Create evaluation plots
    create_evaluation_plots(metrics, total_frames, model)
    
    # Create statistics table
    create_statistics_table(metrics, model, drones, 'evaluation_plots')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved in 'evaluation_plots/' folder:")
    print("  • figure_6_evaluation.png/pdf - Main evaluation figure")
    print("  • behavior_activation.png - Behavior activation plots")
    print("  • distance_histogram.png - Distance histogram")
    print("  • results_summary.txt - Text summary")

if __name__ == "__main__":
    main()