"""
VIDEO_ANIMATION.PY - Creates animation video
Compatible with fixed Drone.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # For video creation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from CreateModel2 import CreateModel2
from Drone import Drone
import os

def run_simulation_for_video(model):
    """Run simulation and collect all data for video"""
    print("Running simulation for video...")
    
    # Initialize drones
    drones = []
    for i in range(1, model['n'] + 1):
        random_offset = np.random.uniform(-1, 1, size=(2,))
        start_pos = np.array(model['start']) + random_offset
        drone = Drone(i, start_pos)
        drones.append(drone)
    
    dt = 0.02
    max_frames = 4000
    reached = False
    
    # Store all positions for animation
    all_positions = [[] for _ in drones]
    
    frame = 0
    while not reached and frame < max_frames:
        reached_sum = 0
        
        for i in range(model['n']):
            # FIXED: Behavior returns 3 values, ignore the third
            vel, reach, _ = drones[i].Behavior(drones, model)
            drones[i].UpdatePosition(vel, dt)
            reached_sum += int(reach)
            
            # Store position for this frame
            all_positions[i].append(drones[i].position.copy())
        
        reached = reached_sum > 0
        frame += 1
        
        if frame % 200 == 0:
            print(f"  Frame {frame}")
    
    total_frames = frame
    print(f"Simulation completed: {total_frames} frames")
    
    return all_positions, total_frames

def create_video_animation(model, positions, total_frames):
    """Create video animation"""
    print(f"\nCreating animation ({total_frames} frames)...")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors
    newcolors = np.array([
        [0, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840]
    ])
    
    # Initialize plot elements
    path_lines = []
    drone_dots = []
    
    for i in range(model['n']):
        line, = ax.plot([], [], linewidth=2, color=newcolors[i % len(newcolors)], alpha=0.7)
        path_lines.append(line)
        
        dot = ax.scatter([], [], marker='o', edgecolor=[0, 0.5, 0.5],
                        facecolor=[0, 0.7, 0.7], linewidth=1.5, s=80)
        drone_dots.append(dot)
    
    # Formation line
    formation_line, = ax.plot([], [], '-k', linewidth=2, alpha=0.5)
    
    # Draw obstacles
    for obs in model['obstacles']:
        polygon = Polygon(obs, closed=True, facecolor='black', alpha=0.7)
        ax.add_patch(polygon)
    
    # Start and goal
    ax.plot(model['start'][0], model['start'][1], 'bs', 
            markersize=10, markerfacecolor='b', label='Start')
    ax.plot(model['goal'][0], model['goal'][1], 'rp', 
            markersize=10, markerfacecolor='r', label='Goal')
    
    # Setup axes
    ax.set_xlim([model['xmin'], model['xmax']])
    ax.set_ylim([model['ymin'], model['ymax']])
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Drone V-Shape Formation Animation')
    
    # Frame info text
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Initialize function
    def init():
        for line in path_lines:
            line.set_data([], [])
        
        for dot in drone_dots:
            dot.set_offsets(np.empty((0, 2)))
        
        formation_line.set_data([], [])
        frame_text.set_text('')
        
        return path_lines + drone_dots + [formation_line, frame_text]
    
    # Update function
    def update(frame_idx):
        for line in path_lines:
            line.set_data([], [])
        
        # Plot paths up to current frame
        for i in range(model['n']):
            if frame_idx < len(positions[i]):
                x_data = [p[0] for p in positions[i][:frame_idx+1]]
                y_data = [p[1] for p in positions[i][:frame_idx+1]]
                path_lines[i].set_data(x_data, y_data)
                
                if len(positions[i]) > frame_idx:
                    current_pos = positions[i][frame_idx]
                    drone_dots[i].set_offsets([current_pos])
        
        # Plot formation
        if frame_idx < len(positions[0]):
            gr = []
            for i in range(model['n']):
                if frame_idx < len(positions[i]):
                    gr.append(positions[i][frame_idx])
            
            if len(gr) > 1:
                gr_array = np.array(gr)
                formation_line.set_data(gr_array[:, 0], gr_array[:, 1])
        
        # Update frame text
        frame_text.set_text(f'Frame: {frame_idx}/{total_frames-1}\nTime: {frame_idx*0.02:.1f}s')
        
        return path_lines + drone_dots + [formation_line, frame_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=min(total_frames, 2000),
                                  init_func=init, blit=True, interval=20)
    
    # Save video
    print("Saving video...")
    
    try:
        # Try FFmpeg
        writer = animation.FFMpegWriter(fps=30, bitrate=5000)
        anim.save('drone_formation_video.mp4', writer=writer, dpi=100)
        print("✓ Video saved as 'drone_formation_video.mp4'")
    except:
        try:
            # Fallback to GIF
            writer = animation.PillowWriter(fps=20)
            anim.save('drone_formation_video.gif', writer=writer, dpi=100)
            print("✓ GIF saved as 'drone_formation_video.gif'")
        except Exception as e:
            print(f"✗ Could not save video: {e}")
            print("Saving individual frames...")
            
            os.makedirs('animation_frames', exist_ok=True)
            for i in range(0, min(total_frames, 100), 10):
                update(i)
                plt.savefig(f'animation_frames/frame_{i:04d}.png', dpi=100)
            
            print("✓ Frames saved in 'animation_frames/' folder")
    
    plt.close(fig)
    return anim

def main():
    """Main function to create video"""
    print("="*60)
    print("DRONE FORMATION VIDEO CREATION")
    print("="*60)
    
    # Create model
    model = CreateModel2()
    
    # Add parameters if not present
    if 'd' not in model:
        model['d'] = 0.8
    if 'alpha' not in model:
        model['alpha'] = 3*np.pi/4
    
    print(f"\nModel Information:")
    print(f"  Number of UAVs: {model['n']}")
    print(f"  Start: {model['start']}")
    print(f"  Goal: {model['goal']}")
    
    # Run simulation
    positions, total_frames = run_simulation_for_video(model)
    
    # Create video
    create_video_animation(model, positions, total_frames)
    
    print("\n" + "="*60)
    print("VIDEO CREATION COMPLETE")
    print("="*60)
    print(f"\nTotal frames simulated: {total_frames}")
    print(f"Total simulation time: {total_frames*0.02:.1f}s")

if __name__ == "__main__":
    main()