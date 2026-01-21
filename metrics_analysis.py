"""
METRICS_ANALYSIS.PY - Tạo các đồ thị đánh giá như trong bài báo
Tương tự Hình 6 trong bài báo SII2024
"""

import numpy as np
import matplotlib.pyplot as plt
from CreateModel2 import CreateModel2
from Drone import Drone
import os
import matplotlib

def run_simulation_with_metrics(model, max_frames=4000):
    """Chạy mô phỏng và thu thập các chỉ số đánh giá"""
    print("Running simulation with metrics collection...")
    
    # Khởi tạo UAVs
    drones = []
    for i in range(1, model['n'] + 1):
        random_offset = np.random.uniform(-1, 1, size=(2,))
        start_pos = np.array(model['start']) + random_offset
        drone = Drone(i, start_pos)
        drones.append(drone)
    
    dt = 0.02
    reached = False
    
    # Lưu trữ dữ liệu để phân tích
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
            vel, reach, behaviors = drones[i].Behavior(drones, model)
            drones[i].UpdatePosition(vel, dt)
            reached_sum += int(reach)
            
            # Lưu vị trí và vận tốc
            all_positions[i].append(drones[i].position.copy())
            all_velocities[i].append(vel.copy())
            
            # Tính heading angle từ vận tốc
            if np.linalg.norm(vel) > 0.001:
                heading = np.arctan2(vel[1], vel[0])
            else:
                heading = 0.0
            all_heading_angles[i].append(heading)
            
            # Đếm số UAV kích hoạt hành vi tái cấu hình và tránh va chạm
            if behaviors.get('reconfig_active', False):
                reconfig_count += 1
            if behaviors.get('collision_avoid_active', False):
                collision_avoid_count += 1
        
        # Tính các chỉ số cho frame hiện tại
        if frame < len(all_positions[0]):
            current_positions = [pos_list[frame] for pos_list in all_positions]
            current_velocities = [vel_list[frame] for vel_list in all_velocities]
            current_headings = [heading_list[frame] for heading_list in all_heading_angles]
            
            # 1. Khoảng cách từng cặp UAV
            pairwise_dists = []
            for i in range(model['n']):
                for j in range(i+1, model['n']):
                    dist = np.linalg.norm(current_positions[i] - current_positions[j])
                    pairwise_dists.append(dist)
            frame_metrics['pairwise_distances'].append(pairwise_dists)
            
            # 2. Sai số trung bình của đội hình
            formation_error = calculate_formation_error(current_positions, model, drones[0])
            frame_metrics['formation_error'].append(formation_error)
            
            # 3. Khoảng cách giữa các UAV liên tiếp
            cons_dists = []
            for i in range(model['n'] - 1):
                dist = np.linalg.norm(current_positions[i] - current_positions[i+1])
                cons_dists.append(dist)
            frame_metrics['consecutive_distances'].append(np.mean(cons_dists) if cons_dists else 0)
            
            # 4. Chỉ số trật tự Φ (Order Metric)
            order_metric = calculate_order_metric(current_headings)
            frame_metrics['order_metric'].append(order_metric)
            
            # 5. Số UAV đang tái cấu hình
            frame_metrics['reconfiguration_count'].append(reconfig_count)
            
            # 6. Số UAV đang tránh va chạm
            frame_metrics['collision_avoidance_count'].append(collision_avoid_count)
            
            # 7. Thời gian
            frame_metrics['time'].append(frame * dt)
        
        reached = reached_sum > 0
        frame += 1
        
        if frame % 500 == 0:
            print(f"  Frame {frame}")
    
    total_frames = frame
    print(f"Simulation completed: {total_frames} frames")
    
    return all_positions, frame_metrics, total_frames, drones

def calculate_formation_error(positions, model, leader_drone):
    """Tính sai số đội hình so với vị trí mong muốn"""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Lấy vị trí leader (UAV ở giữa)
    leader_idx = n // 2
    leader_pos = positions[leader_idx]
    
    # Tính vị trí mong muốn cho các UAV
    desired_positions = []
    for i in range(n):
        if i == leader_idx:
            desired_positions.append(leader_pos)
        else:
            # Tính vị trí mong muốn dựa trên công thức trong bài báo
            d_i = model.get('d', 1.0) * abs(leader_idx - i)
            alpha = model.get('alpha', 3*np.pi/4)
            
            # Xác định UAV ở cánh nào
            if i < leader_idx:
                wing_angle = alpha  # Cánh trái
            else:
                wing_angle = -alpha  # Cánh phải
            
            # Hướng di chuyển của leader (xấp xỉ hướng đến goal)
            if hasattr(leader_drone, 'velocity') and np.linalg.norm(leader_drone.velocity) > 0.1:
                leader_heading = np.arctan2(leader_drone.velocity[1], leader_drone.velocity[0])
            else:
                goal_dir = np.array(model['goal']) - np.array(model['start'])
                leader_heading = np.arctan2(goal_dir[1], goal_dir[0])
            
            desired_pos = leader_pos + d_i * np.array([
                np.cos(leader_heading + wing_angle),
                np.sin(leader_heading + wing_angle)
            ])
            desired_positions.append(desired_pos)
    
    # Tính sai số trung bình (bỏ qua leader)
    errors = []
    for i in range(n):
        if i != leader_idx:
            error = np.linalg.norm(positions[i] - desired_positions[i])
            errors.append(error)
    
    return np.mean(errors) if errors else 0.0

def calculate_order_metric(headings):
    """Tính chỉ số trật tự Φ theo công thức (12)"""
    n = len(headings)
    if n == 0:
        return 0.0
    
    # Tính tổng vector đơn vị
    sum_vector = np.array([0.0, 0.0])
    for angle in headings:
        sum_vector += np.array([np.cos(angle), np.sin(angle)])
    
    # Tính chuẩn và chuẩn hóa
    phi = np.linalg.norm(sum_vector) / n
    
    return phi

def create_evaluation_plots(metrics, total_frames, model, save_path='evaluation_plots'):
    """Tạo 4 đồ thị đánh giá như trong bài báo"""
    
    # Tạo thư mục lưu
    os.makedirs(save_path, exist_ok=True)
    
    # Thời gian (giây)
    time = metrics['time'][:total_frames]
    
    # 1. Đồ thị (a): Khoảng cách giữa các cặp UAV
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Evaluation of the Proposed Algorithm', fontsize=16, fontweight='bold')
    
    # (a) Distances between each pair of UAVs over time
    ax1 = axes[0, 0]
    pairwise_dists = metrics['pairwise_distances'][:total_frames]
    
    if pairwise_dists and len(pairwise_dists[0]) > 0:
        # Vẽ tất cả các cặp
        for pair_idx in range(min(10, len(pairwise_dists[0]))):  # Giới hạn 10 cặp cho rõ
            distances = [dist_list[pair_idx] for dist_list in pairwise_dists 
                        if pair_idx < len(dist_list)]
            ax1.plot(time[:len(distances)], distances, linewidth=1, alpha=0.5)
    
    # Đường alert radius (0.3m) và sensing radius (2.0m)
    ax1.axhline(y=0.3, color='r', linestyle='--', linewidth=2, label='Alert radius (0.3m)')
    ax1.axhline(y=2.0, color='orange', linestyle=':', linewidth=1.5, label='Sensing radius (2.0m)')
    
    # Đường desired distance giữa các UAV liên tiếp
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
    
    # Đường zero error
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
    
    # Đường desired distance (màu đỏ)
    ax3.axhline(y=desired_d, color='r', linestyle='--', linewidth=2, 
                label=f'Desired distance ({desired_d}m)')
    
    # Thêm dải sai số cho phép (±10%)
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
    
    # Đường perfect order và thresholds
    ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Perfect order (Φ=1)')
    ax4.axhline(y=0.9, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Good order (Φ>0.9)')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Order metric Φ')
    ax4.set_title('(d) Values of the order metric Φ over time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0.5, 1.05])
    
    # Đánh dấu các giai đoạn quan trọng nếu có đủ thời gian
    if len(time) > 100:
        # Giai đoạn hình thành (ước lượng)
        formation_time = min(20, time[-1]*0.2)  # 20% thời gian đầu
        ax1.axvline(x=formation_time, color='g', linestyle=':', alpha=0.5, linewidth=1.5)
        ax2.axvline(x=formation_time, color='g', linestyle=':', alpha=0.5, linewidth=1.5)
        ax3.axvline(x=formation_time, color='g', linestyle=':', alpha=0.5, linewidth=1.5)
        ax4.axvline(x=formation_time, color='g', linestyle=':', alpha=0.5, linewidth=1.5)
        
        # Thêm text annotation
        ax1.text(formation_time+1, 2.7, 'Formation\nphase', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # Giai đoạn đi qua đường hẹp (ước lượng giữa simulation)
        if len(time) > 200:
            narrow_passage_time = time[len(time)//2]
            ax1.axvline(x=narrow_passage_time, color='orange', linestyle=':', 
                       alpha=0.5, linewidth=1.5)
            ax2.axvline(x=narrow_passage_time, color='orange', linestyle=':', 
                       alpha=0.5, linewidth=1.5)
            ax3.axvline(x=narrow_passage_time, color='orange', linestyle=':', 
                       alpha=0.5, linewidth=1.5)
            ax4.axvline(x=narrow_passage_time, color='orange', linestyle=':', 
                       alpha=0.5, linewidth=1.5)
            
            ax1.text(narrow_passage_time+1, 0.5, 'Narrow\npassage', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/figure_6_evaluation.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/figure_6_evaluation.pdf', bbox_inches='tight')
    plt.show()
    
    # 2. Tạo đồ thị bổ sung: Số UAV kích hoạt các hành vi
    fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # (a) Số UAV kích hoạt hành vi tái cấu hình
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
    
    # (b) Số UAV kích hoạt hành vi tránh va chạm
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
    plt.show()
    
    # 3. Tạo histogram khoảng cách
    fig3, ax3_1 = plt.subplots(figsize=(10, 6))
    
    # Gom tất cả khoảng cách vào một list
    all_distances = []
    for dist_list in metrics['pairwise_distances'][:total_frames]:
        all_distances.extend(dist_list)
    
    if all_distances:
        ax3_1.hist(all_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax3_1.axvline(x=0.3, color='r', linestyle='--', linewidth=2, label='Alert radius (0.3m)')
        ax3_1.axvline(x=desired_d, color='g', linestyle='--', linewidth=2, 
                     label=f'Desired distance ({desired_d}m)')
        
        # Tính thống kê
        mean_dist = np.mean(all_distances)
        median_dist = np.median(all_distances)
        min_dist = np.min(all_distances)
        
        ax3_1.axvline(x=mean_dist, color='orange', linestyle='-', linewidth=2, 
                     label=f'Mean: {mean_dist:.2f}m')
        
        ax3_1.set_xlabel('Distance between UAVs (m)')
        ax3_1.set_ylabel('Frequency')
        ax3_1.set_title(f'Histogram of distances between UAV pairs (N={len(all_distances)} pairs)')
        ax3_1.grid(True, alpha=0.3)
        ax3_1.legend()
        
        # Hiển thị thông tin thống kê
        stats_text = f'Statistics:\nMean: {mean_dist:.3f} m\nMedian: {median_dist:.3f} m\nMin: {min_dist:.3f} m'
        ax3_1.text(0.95, 0.95, stats_text, transform=ax3_1.transAxes,
                  fontsize=10, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/distance_histogram.png', dpi=300)
    plt.show()
    
    return fig

def create_statistics_table(metrics, model, drones, save_path):
    """Tạo bảng thống kê tương tự TABLE I trong bài báo"""
    
    # Lấy các giá trị thống kê
    formation_errors = metrics['formation_error']
    pairwise_dists_all = metrics['pairwise_distances']
    cons_dists = metrics['consecutive_distances']
    order_metrics = metrics['order_metric']
    
    # Tính toán các chỉ số từ 100 frame cuối (steady state)
    if len(formation_errors) > 100:
        steady_state_start = len(formation_errors) - 100
        avg_error = np.mean(formation_errors[steady_state_start:])
        avg_cons_distance = np.mean(cons_dists[steady_state_start:])
        avg_order_metric = np.mean(order_metrics[steady_state_start:])
    else:
        avg_error = np.mean(formation_errors) if formation_errors else 0
        avg_cons_distance = np.mean(cons_dists) if cons_dists else 0
        avg_order_metric = np.mean(order_metrics) if order_metrics else 0
    
    # Tìm khoảng cách nhỏ nhất giữa các cặp UAV
    min_distances = []
    for dist_list in pairwise_dists_all:
        if dist_list:
            min_distances.append(min(dist_list))
    min_distance = np.min(min_distances) if min_distances else 0
    
    # Tính khoảng cách trung bình giữa tất cả các cặp UAV
    all_pairwise_dists = []
    for dist_list in pairwise_dists_all:
        all_pairwise_dists.extend(dist_list)
    avg_pairwise_distance = np.mean(all_pairwise_dists) if all_pairwise_dists else 0
    
    # Tính tỷ lệ thời gian có UAV tái cấu hình
    reconfig_times = sum(1 for x in metrics['reconfiguration_count'][:len(metrics['time'])] if x > 0)
    reconfig_percentage = (reconfig_times / len(metrics['time'])) * 100 if metrics['time'] else 0
    
    # Tạo bảng
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Dữ liệu bảng - Kết quả hiện tại
    current_results = [
        ["Current Simulation Results", "", "", "", "", "", ""],
        ["Metric", "Value", "Unit", "Target", "Status", "Notes", ""],
        ["Number of UAVs", f"{model['n']}", "UAVs", "-", "OK", f"Scenario with {model['n']} UAVs", ""],
        ["Desired distance (d)", f"{model.get('d', 1.0):.3f}", "m", f"{model.get('d', 1.0):.1f}", "Set", "Distance between consecutive UAVs", ""],
        ["Desired angle (α)", f"{model.get('alpha', 3*np.pi/4)/np.pi:.3f}π", "rad", f"{3/4:.2f}π", "Set", "V-shape wing angle", ""],
        ["Average formation error", f"{avg_error:.5f}", "m", "< 0.15", "✓" if avg_error < 0.15 else "⚠", f"{'Within' if avg_error < 0.15 else 'Above'} target", ""],
        ["Min distance between UAVs", f"{min_distance:.5f}", "m", "> 0.3", "✓" if min_distance > 0.3 else "✗", f"{'Safe' if min_distance > 0.3 else 'Risk of collision'}", ""],
        ["Avg consecutive distance", f"{avg_cons_distance:.5f}", "m", f"{model.get('d', 1.0):.1f}", "✓" if abs(avg_cons_distance - model.get('d', 1.0)) < 0.1 else "⚠", f"Error: {abs(avg_cons_distance - model.get('d', 1.0)):.3f}m", ""],
        ["Order metric (Φ)", f"{avg_order_metric:.3f}", "-", "~1.0", "✓" if avg_order_metric > 0.9 else "⚠", f"{'Good' if avg_order_metric > 0.9 else 'Poor'} alignment", ""],
        ["Reconfiguration activity", f"{reconfig_percentage:.1f}%", "%", "-", "-", f"Active in {reconfig_times}/{len(metrics['time'])} frames", ""],
        ["Simulation duration", f"{metrics['time'][-1]:.1f}", "s", "-", "-", f"{len(metrics['time'])} frames", ""]
    ]
    
    # Dữ liệu từ bài báo để so sánh
    paper_data = [
        ["Comparison with Paper Results (Table I)", "", "", "", "", "", ""],
        ["Scenario", "UAVs", "d", "α", "Avg error", "Min dist", "Avg cons dist"],
        ["1", "3", "1.0", "3π/4", "0.12333", "0.48557", "0.98788"],
        ["2", "5", "1.0", "3π/4", "0.12068", "0.37383", "0.96575"],
        ["3", "3", "0.8", "5π/6", "0.10942", "0.48988", "0.87599"],
        ["4", "3", "1.0", "4π/5", "0.13666", "0.47581", "1.0943"],
        ["5", "5", "0.8", "3π/4", "0.111", "0.41279", "0.88913"]
    ]
    
    # Kết hợp cả hai bảng
    table_data = current_results + [["", "", "", "", "", "", ""]] + paper_data
    
    # Tạo bảng
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    # Tô màu các hàng
    colors = ['#40466e', '#e1e5f1', '#f0f0f0']
    for i in range(len(table_data)):
        for j in range(len(table_data[i])):
            if i == 0 or i == len(current_results) or i == len(current_results) + 1:
                # Tiêu đề
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white', fontsize=10)
            elif i < len(current_results):
                # Dữ liệu hiện tại
                table[(i, j)].set_facecolor(colors[i % 2])
            else:
                # Dữ liệu từ bài báo
                table[(i, j)].set_facecolor('#f5f5f5' if i % 2 == 0 else '#ffffff')
    
    ax.set_title('STATISTICAL EVALUATION: Current Simulation vs. Paper Results', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/statistical_table.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/statistical_table.pdf', bbox_inches='tight')
    plt.show()
    
    # In ra console
    print("\n" + "="*100)
    print("STATISTICAL EVALUATION RESULTS")
    print("="*100)
    print(f"{'Metric':<30} {'Value':<15} {'Unit':<10} {'Target':<15} {'Status':<10}")
    print("-"*100)
    print(f"{'Number of UAVs':<30} {model['n']:<15} {'UAVs':<10} {'-':<15} {'OK':<10}")
    print(f"{'Desired distance (d)':<30} {model.get('d', 1.0):<15.3f} {'m':<10} {model.get('d', 1.0):<15.1f} {'Set':<10}")
    print(f"{'Desired angle (α)':<30} {model.get('alpha', 3*np.pi/4)/np.pi:<15.3f}π {'rad':<10} {3/4:<15.2f}π {'Set':<10}")
    print(f"{'Average formation error':<30} {avg_error:<15.5f} {'m':<10} {'< 0.15':<15} {'✓' if avg_error < 0.15 else '⚠':<10}")
    print(f"{'Min distance between UAVs':<30} {min_distance:<15.5f} {'m':<10} {'> 0.3':<15} {'✓' if min_distance > 0.3 else '✗':<10}")
    print(f"{'Avg consecutive distance':<30} {avg_cons_distance:<15.5f} {'m':<10} {model.get('d', 1.0):<15.1f} {'✓' if abs(avg_cons_distance - model.get('d', 1.0)) < 0.1 else '⚠':<10}")
    print(f"{'Order metric (Φ)':<30} {avg_order_metric:<15.3f} {'-':<10} {'~1.0':<15} {'✓' if avg_order_metric > 0.9 else '⚠':<10}")
    print(f"{'Reconfiguration activity':<30} {reconfig_percentage:<15.1f} {'%':<10} {'-':<15} {'-':<10}")
    print(f"{'Simulation duration':<30} {metrics['time'][-1]:<15.1f} {'s':<10} {'-':<15} {'-':<10}")
    print("="*100)
    
    # Lưu kết quả ra file text
    with open(f'{save_path}/results_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMULATION RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Simulation Parameters:\n")
        f.write(f"  Number of UAVs: {model['n']}\n")
        f.write(f"  Desired distance d: {model.get('d', 1.0):.3f}m\n")
        f.write(f"  Desired angle α: {model.get('alpha', 3*np.pi/4)/np.pi:.3f}π rad\n")
        f.write(f"  Simulation duration: {metrics['time'][-1]:.1f}s ({len(metrics['time'])} frames)\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Average formation error: {avg_error:.5f}m\n")
        f.write(f"  Minimum inter-UAV distance: {min_distance:.5f}m\n")
        f.write(f"  Average consecutive UAV distance: {avg_cons_distance:.5f}m\n")
        f.write(f"  Average order metric Φ: {avg_order_metric:.3f}\n")
        f.write(f"  Reconfiguration activity: {reconfig_percentage:.1f}%\n\n")
        
        f.write("Safety Assessment:\n")
        f.write(f"  Collision avoidance: {'✓ SAFE' if min_distance > 0.3 else '✗ UNSAFE'}\n")
        f.write(f"  Formation stability: {'✓ STABLE' if avg_error < 0.15 else '⚠ UNSTABLE'}\n")
        f.write(f"  UAV alignment: {'✓ GOOD' if avg_order_metric > 0.9 else '⚠ POOR'}\n")

def main():
    """Hàm chính để chạy đánh giá"""
    print("="*80)
    print("DRONE FORMATION METRICS EVALUATION SYSTEM")
    print("(Generating plots similar to Figure 6 in the paper)")
    print("="*80)
    
    # Tạo model
    model = CreateModel2()
    
    # Thêm thông số V-shape vào model nếu chưa có
    if 'd' not in model:
        model['d'] = 0.8  # Khoảng cách mong muốn (m)
    if 'alpha' not in model:
        model['alpha'] = 3*np.pi/4  # Góc cánh (rad)
    
    print(f"\n📊 Model parameters:")
    print(f"   • Number of UAVs: {model['n']}")
    print(f"   • Desired distance d: {model['d']}m")
    print(f"   • Desired angle α: {model['alpha']/np.pi:.2f}π rad ({model['alpha']:.2f} rad)")
    print(f"   • Alert radius: 0.3m")
    print(f"   • Sensing radius: 2.0m")
    print(f"   • Start position: {model['start']}")
    print(f"   • Goal position: {model['goal']}")
    
    print(f"\n🚀 Running simulation with metrics collection...")
    
    # Chạy mô phỏng và thu thập metrics
    positions, metrics, total_frames, drones = run_simulation_with_metrics(model, max_frames=4000)
    
    print(f"\n✅ Simulation completed successfully!")
    print(f"   • Total frames: {total_frames}")
    print(f"   • Total time: {total_frames*0.02:.1f}s")
    print(f"   • Data collected: {len(metrics['time'])} time points")
    
    print(f"\n📈 Generating evaluation plots...")
    
    # Tạo các đồ thị đánh giá
    create_evaluation_plots(metrics, total_frames, model)
    
    # Tạo bảng thống kê
    create_statistics_table(metrics, model, drones, 'evaluation_plots')
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved in 'evaluation_plots/' folder:")
    print(f"   └── figure_6_evaluation.png/pdf     - Main evaluation figure (4 subplots)")
    print(f"   └── behavior_activation.png         - Behavior activation over time")
    print(f"   └── distance_histogram.png          - Histogram of inter-UAV distances")
    print(f"   └── statistical_table.png/pdf       - Statistical comparison table")
    print(f"   └── results_summary.txt             - Text summary of results")
    print("\n📊 Key metrics available in console output above.")
    print("="*80)

if __name__ == "__main__":
    main()