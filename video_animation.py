"""
VIDEO_ANIMATION.PY - Equivalent of MATLAB Video.m
Creates animation video like MATLAB
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
    
    # Initialize drones like MATLAB main.m
    drones = []
    for i in range(1, model['n'] + 1):
        random_offset = np.random.uniform(-1, 1, size=(2,))
        start_pos = np.array(model['start']) + random_offset
        drone = Drone(i, start_pos)
        drones.append(drone)
    
    dt = 0.02
    max_frames = 4000  # ĐÃ TĂNG GẤP ĐÔI: từ 2000 lên 4000 frames (80 giây)
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
            # Debug: in vị trí UAV và khoảng cách đến goal
            if drones and frame % 400 == 0:  # Mỗi 400 frames in một lần
                uav_pos = drones[0].position
                goal_pos = np.array(model['goal'])
                distance = np.linalg.norm(uav_pos - goal_pos)
                print(f"    UAV1 position: [{uav_pos[0]:.1f}, {uav_pos[1]:.1f}]")
                print(f"    Goal position: [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}]")
                print(f"    Distance to goal: {distance:.1f}m")
                
                # Kiểm tra xem UAV có đi qua đường hầm không
                if 'obstacles' in model and len(model['obstacles']) > 0:
                    # Giả sử obstacle đầu tiên là đường hầm
                    tunnel = model['obstacles'][0]
                    tunnel_x_min = min([p[0] for p in tunnel])
                    tunnel_x_max = max([p[0] for p in tunnel])
                    
                    if uav_pos[0] > tunnel_x_min and uav_pos[0] < tunnel_x_max:
                        print(f"    UAV đang trong đường hầm (x={uav_pos[0]:.1f})")
                    elif uav_pos[0] >= tunnel_x_max:
                        print(f"    UAV đã vượt qua đường hầm!")
    
    total_frames = frame
    print(f"Simulation completed: {total_frames} frames")
    
    if frame >= max_frames:
        print("⚠️  WARNING: Max frames reached before all UAVs reached goal!")
        print(f"   Last UAV1 position: {drones[0].position}")
        print(f"   Goal position: {model['goal']}")
        print("   Consider increasing max_frames or checking drone behavior.")
    else:
        print("✓ All UAVs reached the goal!")
    
    return all_positions, total_frames

def create_video_animation(model, positions, total_frames):
    """Create video animation like MATLAB Video.m"""
    print(f"\nCreating animation ({total_frames} frames)...")
    
    # Setup figure like MATLAB
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # MATLAB colors
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
        # Path line
        line, = ax.plot([], [], linewidth=2, color=newcolors[i % len(newcolors)], 
                       alpha=0.7)
        path_lines.append(line)
        
        # Drone dot
        dot = ax.scatter([], [], marker='o',
                        edgecolor=[0, 0.5, 0.5],
                        facecolor=[0, 0.7, 0.7],
                        linewidth=1.5, s=80)
        drone_dots.append(dot)
    
    # Formation line
    formation_line, = ax.plot([], [], '-k', linewidth=2, alpha=0.5)
    
    # Draw obstacles (like MATLAB PlotResults.m)
    for obs in model['obstacles']:
        polygon = Polygon(obs, closed=True, facecolor='black', alpha=0.7)
        ax.add_patch(polygon)
    
    # Start and goal (like MATLAB)
    ax.plot(model['start'][0], model['start'][1], 'bs', 
            markersize=10, markerfacecolor='b', label='Start')
    ax.plot(model['goal'][0], model['goal'][1], 'rp', 
            markersize=10, markerfacecolor='r', label='Goal')
    
    # Thêm điểm đánh dấu giữa đường hầm (nếu cần)
    if 'obstacles' in model and len(model['obstacles']) > 0:
        tunnel = model['obstacles'][0]
        tunnel_center_x = sum([p[0] for p in tunnel]) / len(tunnel)
        tunnel_center_y = sum([p[1] for p in tunnel]) / len(tunnel)
        ax.plot(tunnel_center_x, tunnel_center_y, 'gx', 
                markersize=8, markeredgewidth=2, label='Tunnel Center')
    
    # Setup axes
    ax.set_xlim([model['xmin'], model['xmax']])
    ax.set_ylim([model['ymin'], model['ymax']])
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Drone V-Shape Formation Animation - Extended Simulation')
    
    # Frame info text
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Progress bar
    progress_bar = ax.text(0.02, 0.92, '', transform=ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Initialize function
    def init():
        for line in path_lines:
            line.set_data([], [])
        
        for dot in drone_dots:
            dot.set_offsets(np.empty((0, 2)))
        
        formation_line.set_data([], [])
        frame_text.set_text('')
        progress_bar.set_text('')
        
        return path_lines + drone_dots + [formation_line, frame_text, progress_bar]
    
    # Update function (like MATLAB Video.m loop)
    def update(frame_idx):
        # Clear axis like MATLAB: cla;
        for line in path_lines:
            line.set_data([], [])
        
        # Plot paths up to current frame
        for i in range(model['n']):
            if frame_idx < len(positions[i]):
                # Get positions up to this frame
                x_data = [p[0] for p in positions[i][:frame_idx+1]]
                y_data = [p[1] for p in positions[i][:frame_idx+1]]
                path_lines[i].set_data(x_data, y_data)
                
                # Update drone position
                if len(positions[i]) > frame_idx:
                    current_pos = positions[i][frame_idx]
                    drone_dots[i].set_offsets([current_pos])
        
        # Plot formation graph at current frame (like MATLAB)
        if frame_idx < len(positions[0]):
            gr = []
            for i in range(model['n']):
                if frame_idx < len(positions[i]):
                    gr.append(positions[i][frame_idx])
            
            if len(gr) > 1:
                gr_array = np.array(gr)
                formation_line.set_data(gr_array[:, 0], gr_array[:, 1])
        
        # Update frame text
        frame_text.set_text(f'Frame: {frame_idx}/{total_frames-1}\n'
                          f'Time: {frame_idx*0.02:.1f}s')
        
        # Update progress bar
        progress_percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
        progress_bar.set_text(f'Progress: {progress_percent:.1f}%')
        
        return path_lines + drone_dots + [formation_line, frame_text, progress_bar]
    
    # Tạo animation với tất cả frames (không giới hạn 1000 frames nữa)
    anim = animation.FuncAnimation(fig, update, frames=total_frames,
                                  init_func=init, blit=True, interval=20)
    
    # Save video (like MATLAB VideoWriter)
    print("Saving video...")
    
    try:
        # Try FFmpeg first for MP4
        writer = animation.FFMpegWriter(fps=30, bitrate=5000)
        output_file = 'drone_formation_video_extended.mp4'
        anim.save(output_file, writer=writer, dpi=100, 
                  progress_callback=lambda i, n: print(f"  Encoding frame {i}/{n}", end='\r'))
        print(f"\n✓ Video saved as '{output_file}'")
        print(f"  Duration: {total_frames*0.02:.1f}s, Frames: {total_frames}")
        
    except Exception as e1:
        print(f"\n✗ MP4 creation failed: {e1}")
        print("Trying GIF format...")
        
        try:
            # Fallback to GIF (sẽ lâu hơn và file lớn hơn)
            writer = animation.PillowWriter(fps=20)
            output_file = 'drone_formation_video_extended.gif'
            anim.save(output_file, writer=writer, dpi=80,
                      progress_callback=lambda i, n: print(f"  Encoding frame {i}/{n}", end='\r'))
            print(f"\n✓ GIF saved as '{output_file}'")
            print("  Note: GIF files are large. For better quality, install FFmpeg.")
            
        except Exception as e2:
            print(f"\n✗ GIF creation failed: {e2}")
            print("Saving individual frames...")
            
            # Save key frames
            os.makedirs('animation_frames', exist_ok=True)
            save_interval = max(1, total_frames // 100)  # Lưu 100 frames
            for i in range(0, total_frames, save_interval):
                update(i)
                plt.savefig(f'animation_frames/frame_{i:05d}.png', dpi=100, bbox_inches='tight')
                if i % (save_interval * 10) == 0:
                    print(f"  Saved frame {i}/{total_frames}")
            
            print(f"\n✓ {total_frames//save_interval} frames saved in 'animation_frames/' folder")
            print("  You can create video using: ffmpeg -framerate 30 -i frame_%05d.png output.mp4")
    
    plt.close(fig)
    return anim

def main():
    """Main function to create video"""
    print("="*60)
    print("DRONE FORMATION VIDEO CREATION - EXTENDED SIMULATION")
    print("="*60)
    
    # Create model
    model = CreateModel2()
    
    # Debug: in thông tin model
    print(f"\nModel Information:")
    print(f"  Number of UAVs: {model['n']}")
    print(f"  Start position: {model['start']}")
    print(f"  Goal position: {model['goal']}")
    print(f"  Map bounds: x[{model['xmin']}, {model['xmax']}], y[{model['ymin']}, {model['ymax']}]")
    
    if 'obstacles' in model and len(model['obstacles']) > 0:
        print(f"  Number of obstacles: {len(model['obstacles'])}")
        # Kiểm tra kích thước đường hầm (obstacle đầu tiên)
        tunnel = model['obstacles'][0]
        tunnel_x_min = min([p[0] for p in tunnel])
        tunnel_x_max = max([p[0] for p in tunnel])
        tunnel_y_min = min([p[1] for p in tunnel])
        tunnel_y_max = max([p[1] for p in tunnel])
        print(f"  Tunnel dimensions: {tunnel_x_max-tunnel_x_min:.1f}m x {tunnel_y_max-tunnel_y_min:.1f}m")
        print(f"  Tunnel position: x[{tunnel_x_min:.1f}, {tunnel_x_max:.1f}]")
    
    print(f"\nSimulation parameters:")
    print(f"  Max frames: 4000")
    print(f"  Time step: 0.02s")
    print(f"  Max simulation time: {4000*0.02:.1f}s")
    
    # Run simulation
    positions, total_frames = run_simulation_for_video(model)
    
    # Create video
    create_video_animation(model, positions, total_frames)
    
    print("\n" + "="*60)
    print("VIDEO CREATION COMPLETE")
    print("="*60)
    
    # Hiển thị thông tin tổng kết
    print(f"\nSummary:")
    print(f"  Total frames simulated: {total_frames}")
    print(f"  Total simulation time: {total_frames*0.02:.1f}s")
    if total_frames < 4000:
        print(f"  Simulation finished early: All UAVs reached goal")
    else:
        print(f"  Simulation hit max frames: Consider increasing max_frames")

if __name__ == "__main__":
    main()