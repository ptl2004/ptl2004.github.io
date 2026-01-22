"""
VIDEO_ANIMATION.PY - Creates animation video
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
import warnings

def run_simulation_for_video(model, max_frames=1000):
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
    reached = False
    
    # Store all positions for animation
    all_positions = [[] for _ in drones]
    
    frame = 0
    while not reached and frame < max_frames:
        reached_sum = 0
        
        for i in range(model['n']):
            vel, reach = drones[i].Behavior(drones, model)
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
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
    
    # Initialize plot elements
    path_lines = []
    drone_dots = []
    
    for i in range(model['n']):
        line, = ax.plot([], [], linewidth=2, 
                       color=colors[i % len(colors)], 
                       alpha=0.7)
        path_lines.append(line)
        
        dot = ax.scatter([], [], marker='o', edgecolor='darkblue',
                        facecolor='lightblue', linewidth=1.5, s=80)
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
        # Plot paths up to current frame
        for i in range(model['n']):
            if frame_idx < len(positions[i]):
                x_data = [p[0] for p in positions[i][:frame_idx+1]]
                y_data = [p[1] for p in positions[i][:frame_idx+1]]
                
                if len(x_data) > 0:
                    path_lines[i].set_data(x_data, y_data)
                
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
        frame_text.set_text(f'Frame: {frame_idx+1}/{total_frames}\nTime: {(frame_idx+1)*0.02:.1f}s')
        
        return path_lines + drone_dots + [formation_line, frame_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=min(total_frames, 3000),
                                  init_func=init, blit=True, interval=20)
    
    # Save video - THỬ NHIỀU CÁCH
    print("Saving animation...")
    
    # 1. Thử tạo GIF trước (dễ nhất)
    try:
        writer = animation.PillowWriter(fps=20)
        gif_path = 'drone_formation_animation.gif'
        anim.save(gif_path, writer=writer, dpi=100)
        print(f"✓ GIF saved as '{gif_path}'")
        return gif_path
    except Exception as e:
        print(f"✗ Could not save GIF: {e}")
    
    # 2. Thử tạo MP4 nếu có ffmpeg
    try:
        writer = animation.FFMpegWriter(fps=30, bitrate=5000)
        mp4_path = 'drone_formation_video.mp4'
        anim.save(mp4_path, writer=writer, dpi=100)
        print(f"✓ MP4 saved as '{mp4_path}'")
        return mp4_path
    except Exception as e:
        print(f"✗ Could not save MP4: {e}")
    
    # 3. Lưu từng frame riêng lẻ
    print("Saving individual frames...")
    
    frames_dir = 'animation_frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Chỉ lưu 50 frame để tránh quá nhiều file
    save_every = max(1, total_frames // 50)
    for i in range(0, total_frames, save_every):
        update(i)
        frame_file = os.path.join(frames_dir, f'frame_{i:04d}.png')
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
    
    print(f"✓ Frames saved in '{frames_dir}/' folder")
    
    # Lưu ảnh tổng hợp
    update(total_frames-1)  # Frame cuối cùng
    plt.savefig('formation_final_frame.png', dpi=300, bbox_inches='tight')
    print("✓ Final frame saved as 'formation_final_frame.png'")
    
    plt.close(fig)
    return frames_dir

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
    print(f"  Desired distance: {model['d']} m")
    print(f"  Desired angle: {model['alpha']:.2f} rad")
    print(f"  Start: {model['start']}")
    print(f"  Goal: {model['goal']}")
    
    # Run simulation
    positions, total_frames = run_simulation_for_video(model, max_frames=3000)
    
    # Create animation
    result_path = create_video_animation(model, positions, total_frames)
    
    print("\n" + "="*60)
    print("ANIMATION CREATION COMPLETE")
    print("="*60)
    print(f"\nTotal frames simulated: {total_frames}")
    print(f"Total simulation time: {total_frames*0.02:.1f}s")
    
    if result_path:
        print(f"Animation saved at: {result_path}")
    
    return result_path

if __name__ == "__main__":
    main()