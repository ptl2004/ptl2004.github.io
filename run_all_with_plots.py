"""
RUN_SIMULATION_AND_PLOT.PY - Chạy simulation và vẽ đồ thị đánh giá
Đồ án tốt nghiệp: Self-Reconfigurable V-shape Formation of Multiple UAVs
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec
import sys
import os

# Import các module gốc
sys.path.append('.')

def run_complete_simulation():
    """Chạy simulation thực sự và trả về dữ liệu"""
    print("="*70)
    print("KHOI TAO SIMULATION UAV FORMATION")
    print("="*70)
    
    # Import modules
    from CreateModel2 import CreateModel2
    from Drone import Drone
    
    # Tạo model
    model = CreateModel2()
    
    print("Thong so mo hinh:")
    print("  - So UAV: %d" % model['n'])
    print("  - Khoang cach mong muon: d = %.2f m" % model['d'])
    print("  - Goc doi hinh: alpha = %.2f rad" % model['alpha'])
    print("  - Vi tri bat dau: %s" % str(model['start']))
    print("  - Vi tri dich: %s" % str(model['goal']))
    
    # Khởi tạo UAVs
    drones = []
    print("\nKhoi tao UAVs voi vi tri ngau nhien:")
    for i in range(1, model['n'] + 1):
        random_offset = np.random.uniform(-1, 1, size=(2,))
        start_pos = np.array(model['start']) + random_offset
        drone = Drone(i, start_pos)
        drones.append(drone)
        print("  UAV%d: [%.2f, %.2f]" % (i, start_pos[0], start_pos[1]))
    
    # Chạy simulation
    dt = 0.02
    iter_count = 0
    reached = False
    max_iterations = 2000
    
    print("\nBat dau simulation (dt=%.2fs, toi da=%d vong lap)..." % (dt, max_iterations))
    
    while not reached and iter_count < max_iterations:
        reached_sum = 0
        
        for i in range(model['n']):
            vel, reach = drones[i].Behavior(drones, model)
            reached_sum += int(reach)
            drones[i].UpdatePosition(vel, dt)
        
        iter_count += 1
        
        # Hiển thị tiến độ
        if iter_count % 200 == 0:
            leader_idx = (model['n'] + 1) // 2 - 1
            leader_pos = drones[leader_idx].position
            goal_dist = np.linalg.norm(np.array(model['goal']) - leader_pos)
            print("  Vong lap %d: Khoang cach toi dich = %.2f m" % (iter_count, goal_dist))
        
        # Điều kiện dừng
        reached = reached_sum > 0
    
    if iter_count >= max_iterations:
        print("\nCANH BAO: Dat gioi han %d vong lap (chua den dich)" % max_iterations)
        print("   Vi tri cuoi cung: UAV1 = %s" % str(drones[0].position))
        print("   Vi tri dich: %s" % str(model['goal']))
    else:
        print("\nTHANH CONG: Simulation hoan thanh sau %d vong lap" % iter_count)
        print("   Thoi gian: %.2f giay" % (iter_count * dt))
    
    # Tính quãng đường di chuyển
    total_distance = 0
    for i, drone in enumerate(drones):
        if len(drone.path) > 1:
            path_array = np.array(drone.path)
            distance = np.sum(np.linalg.norm(np.diff(path_array[:, :2], axis=0), axis=1))
            total_distance += distance
            print("  UAV%d: %.2f m" % (i+1, distance))
    
    print("  Tong quang duong: %.2f m" % total_distance)
    
    return drones, model, iter_count

def calculate_metrics(drones, model):
    """Tính toán metrics từ dữ liệu simulation"""
    print("\n" + "="*70)
    print("TINH TOAN METRICS CHO DO THI")
    print("="*70)
    
    n_steps = len(drones[0].path)
    n_drones = model['n']
    
    print("Dang xu ly %d buoc thoi gian cho %d UAV..." % (n_steps, n_drones))
    
    # 1. Khoảng cách giữa các cặp UAV
    print("  [1/4] Tinh khoang cach giua cac cap UAV...")
    pair_distances = {}
    
    for i in range(n_drones):
        for j in range(i + 1, n_drones):
            distances = []
            for k in range(n_steps):
                pos_i = drones[i].path[k][:2]
                pos_j = drones[j].path[k][:2]
                dist = np.linalg.norm(pos_i - pos_j)
                distances.append(dist)
            
            pair_name = 'UAV%d-UAV%d' % (i+1, j+1)
            pair_distances[pair_name] = distances
    
    # 2. Sai số trung bình
    print("  [2/4] Tinh sai so doi hinh...")
    avg_errors = []
    desired_dist = model['d']
    
    for k in range(n_steps):
        errors = []
        for i in range(n_drones - 1):
            pos_i = drones[i].path[k][:2]
            pos_j = drones[i+1].path[k][:2]
            actual_dist = np.linalg.norm(pos_i - pos_j)
            error = abs(actual_dist - desired_dist)
            errors.append(error)
        
        avg_error = np.mean(errors) if errors else 0.0
        avg_errors.append(avg_error)
    
    # 3. Khoảng cách UAV liên tiếp
    print("  [3/4] Tinh khoang cach UAV lien tiep...")
    consecutive_distances = []
    for i in range(n_drones - 1):
        distances = []
        for k in range(n_steps):
            pos_i = drones[i].path[k][:2]
            pos_j = drones[i+1].path[k][:2]
            distances.append(np.linalg.norm(pos_i - pos_j))
        consecutive_distances.append(distances)
    
    # 4. Order metric Phi
    print("  [4/4] Tinh order metric Phi...")
    phi_values = []
    for k in range(n_steps):
        heading_vec = np.array([0.0, 0.0])
        for i in range(n_drones):
            heading_angle = drones[i].path[k][2]
            heading_vec += np.array([np.cos(heading_angle), np.sin(heading_angle)])
        
        phi = np.linalg.norm(heading_vec) / n_drones
        phi_values.append(phi)
    
    # Thời gian
    time_steps = np.arange(n_steps) * 0.02
    
    metrics = {
        'pair_distances': pair_distances,
        'avg_errors': avg_errors,
        'consecutive_distances': consecutive_distances,
        'phi_values': phi_values,
        'time_steps': time_steps,
        'n_steps': n_steps,
        'n_drones': n_drones
    }
    
    print("THANH CONG: Hoan thanh tinh toan metrics")
    return metrics

def plot_figure_6(metrics, model):
    """Vẽ 4 đồ thị như Hình 6 trong bài báo"""
    print("\n" + "="*70)
    print("VE 4 DO THI NHU HINH 6 TRONG BAI BAO")
    print("="*70)
    
    # Tạo figure
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Màu sắc
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange', 'black']
    
    # ===== SUBPLOT 1: Distances between UAVs =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Alert radius line
    ax1.axhline(y=0.3, color='red', linestyle='--', 
                linewidth=1.5, label='Ban kinh canh bao (0.3 m)', alpha=0.7)
    
    # Plot each pair
    pair_names = list(metrics['pair_distances'].keys())
    for idx, pair_name in enumerate(pair_names):
        distances = metrics['pair_distances'][pair_name]
        time_steps = metrics['time_steps'][:len(distances)]
        
        ax1.plot(time_steps, distances, 
                color=colors[idx % len(colors)],
                linewidth=1.2,
                alpha=0.8,
                label=pair_name)
    
    ax1.set_xlabel('Thoi gian (s)', fontsize=11)
    ax1.set_ylabel('Khoang cach (m)', fontsize=11)
    ax1.set_title('(a) Khoang cach giua cac cap UAV theo thoi gian', 
                 fontsize=12, pad=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.set_xlim([0, metrics['time_steps'][-1]])
    ax1.set_ylim([0, 3.0])
    
    # ===== SUBPLOT 2: Average distance error =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    avg_errors = metrics['avg_errors']
    time_steps = metrics['time_steps'][:len(avg_errors)]
    
    ax2.plot(time_steps, avg_errors, 
            color='purple',
            linewidth=2.0,
            alpha=0.9)
    
    ax2.fill_between(time_steps, 0, avg_errors, 
                     color='purple', alpha=0.15)
    
    ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Thoi gian (s)', fontsize=11)
    ax2.set_ylabel('Sai so trung binh (m)', fontsize=11)
    ax2.set_title('(b) Sai so trung binh cua doi hinh UAV', 
                 fontsize=12, pad=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, metrics['time_steps'][-1]])
    ax2.set_ylim([0, 0.15])
    
    # ===== SUBPLOT 3: Distance between consecutive UAVs =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    box_data = metrics['consecutive_distances']
    labels = ['UAV%d-UAV%d' % (i+1, i+2) for i in range(len(box_data))]
    
    bp = ax3.boxplot(box_data, labels=labels, patch_artist=True,
                     widths=0.6, showfliers=False)
    
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Desired distance line
    desired_d = model['d']
    ax3.axhline(y=desired_d, color='red', linestyle='--', 
                linewidth=2.0, label='Khoang cach mong muon (%.1f m)' % desired_d)
    
    ax3.set_ylabel('Khoang cach (m)', fontsize=11)
    ax3.set_title('(c) Khoang cach giua cac UAV lien tiep', 
                 fontsize=12, pad=10, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim([0.2, 1.4])
    
    # ===== SUBPLOT 4: Order metric Phi =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    phi_values = metrics['phi_values']
    time_steps = metrics['time_steps'][:len(phi_values)]
    
    ax4.plot(time_steps, phi_values, 
            color='green',
            linewidth=2.5,
            alpha=0.9,
            label='Order metric Phi')
    
    ax4.fill_between(time_steps, 0.95, phi_values, 
                     where=(np.array(phi_values) >= 0.95),
                     color='lime', alpha=0.2, label='Do dong bo cao (Phi > 0.95)')
    
    ax4.axhline(y=0.95, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Do dong bo hoan hao (Phi = 1.0)')
    
    ax4.set_xlabel('Thoi gian (s)', fontsize=11)
    ax4.set_ylabel('Order metric Phi', fontsize=11)
    ax4.set_title('(d) Gia tri order metric Phi theo thoi gian', 
                 fontsize=12, pad=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, metrics['time_steps'][-1]])
    ax4.set_ylim([0.7, 1.05])
    ax4.legend(loc='lower right', fontsize=9)
    
    # Thêm thông tin
    info_text = ('Ket qua simulation:\n'
                 '- So UAV: %d, d=%.1fm, alpha=%.2frad\n'
                 '- Thoi gian: %.1fs, So buoc: %d\n'
                 '- Ban kinh canh bao: 0.3m, Ban kinh cam bien: 2.0m' % 
                 (model['n'], model['d'], model['alpha'], 
                  metrics['time_steps'][-1], metrics['n_steps']))
    
    fig.text(0.02, 0.02, info_text, 
             fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.suptitle('Danh gia thuat toan dieu khien doi hinh hinh chu V (Hinh 6 - Bai bao)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Lưu file
    output_file = 'figure_6_paper_style.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("THANH CONG: Da luu 4 do thi vao: %s" % output_file)
    
    plt.show()
    
    return fig

def plot_trajectory(drones, model):
    """Vẽ đường đi của UAVs"""
    print("\n" + "="*70)
    print("VE DUONG DI CUA UAVs")
    print("="*70)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Start và Goal
    ax.plot(model['start'][0], model['start'][1], 'bs', 
            markersize=10, markerfacecolor='b', label='Diem bat dau')
    ax.plot(model['goal'][0], model['goal'][1], 'rp', 
            markersize=10, markerfacecolor='r', label='Diem dich')
    
    # Vật cản
    for obs in model['obstacles']:
        polygon = Polygon(obs, closed=True, facecolor='black', alpha=0.7)
        ax.add_patch(polygon)
    
    # Đường đi
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(model['n']):
        path_array = np.array(drones[i].path)
        ax.plot(path_array[:, 0], path_array[:, 1], 
                color=colors[i % len(colors)], 
                linewidth=2, 
                label='UAV%d' % (i+1))
    
    # Vẽ formation tại vài thời điểm
    if len(drones[0].path) > 0:
        num_points = 5
        step = max(1, len(drones[0].path) // num_points)
        
        for i in range(num_points + 1):
            idx = i * step
            if idx >= len(drones[0].path):
                idx = len(drones[0].path) - 1
            
            # Nối các UAV
            points = []
            for j in range(model['n']):
                points.append(drones[j].path[idx][:2])
            
            points_array = np.array(points)
            ax.plot(points_array[:, 0], points_array[:, 1], 
                   '-k', linewidth=1, alpha=0.5)
            
            # Vẽ UAVs
            for j in range(model['n']):
                ax.scatter(drones[j].path[idx][0], drones[j].path[idx][1],
                          marker='o',
                          edgecolor='darkblue',
                          facecolor='lightblue',
                          linewidth=1,
                          s=60, alpha=0.7)
    
    ax.set_xlabel('Toa do x (m)', fontsize=12)
    ax.set_ylabel('Toa do y (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim([model['xmin'], model['xmax']])
    ax.set_ylim([model['ymin'], model['ymax']])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title('Duong di doi hinh hinh chu V trong khong gian hep', fontsize=14)
    
    plt.tight_layout()
    
    # Lưu file
    output_file = 'formation_trajectory.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print("THANH CONG: Da luu duong di vao: %s" % output_file)
    
    plt.show()
    
    return fig

def print_statistics(metrics, model):
    """In thống kê kết quả"""
    print("\n" + "="*70)
    print("THONG KE KET QUA")
    print("="*70)
    
    # Sai số trung bình
    avg_error = np.mean(metrics['avg_errors'])
    max_error = np.max(metrics['avg_errors'])
    
    # Khoảng cách tối thiểu
    min_dist = float('inf')
    for dist_list in metrics['pair_distances'].values():
        current_min = min(dist_list)
        if current_min < min_dist:
            min_dist = current_min
    
    # Order metric
    phi_mean = np.mean(metrics['phi_values'])
    phi_min = np.min(metrics['phi_values'])
    
    print("\nKET QUA CHINH:")
    print("  1. Sai so doi hinh trung binh: %.4f m" % avg_error)
    print("  2. Sai so lon nhat: %.4f m" % max_error)
    print("  3. Khoang cach UAV gan nhat: %.4f m" % min_dist)
    print("  4. Order metric Phi trung binh: %.4f" % phi_mean)
    print("  5. Order metric Phi nho nhat: %.4f" % phi_min)
    
    # Kiểm tra an toàn
    print("\nKIEM TRA AN TOAN:")
    if min_dist > 0.3:
        print("  AN TOAN: Tat ca UAV cach nhau hon 0.3 m")
    else:
        print("  CANH BAO: Co UAV cach nhau <= 0.3 m")
    
    if phi_mean > 0.9:
        print("  TOT: Doi hinh dong bo cao (Phi > 0.9)")
    else:
        print("  CANH BAO: Doi hinh kem dong bo")
    
    print("\nTHONG TIN SIMULATION:")
    print("  - So buoc thoi gian: %d" % metrics['n_steps'])
    print("  - Tong thoi gian: %.2f s" % metrics['time_steps'][-1])
    print("  - So cap UAV: %d" % len(metrics['pair_distances']))
    
    print("\n" + "="*70)

def main():
    """Hàm chính - chạy tất cả"""
    print("="*80)
    print("DO AN TOT NGHIEP: DIEU KHIEN DOI HINH HINH CHU V CHO NHIEU UAV")
    print("="*80)
    
    try:
        # 1. Chạy simulation
        drones, model, iterations = run_complete_simulation()
        
        # 2. Vẽ đường đi
        plot_trajectory(drones, model)
        
        # 3. Tính metrics
        metrics = calculate_metrics(drones, model)
        
        # 4. Vẽ 4 đồ thị như bài báo
        plot_figure_6(metrics, model)
        
        # 5. In thống kê
        print_statistics(metrics, model)
        
        print("\nHOAN THANH! Da tao day du do thi nhu bai bao.")
        print("Cac file da tao:")
        print("  - formation_trajectory.png - Duong di cua UAVs")
        print("  - figure_6_paper_style.png - 4 do thi danh gia")
        
    except KeyboardInterrupt:
        print("\n\nDA DUNG BOI NGUOI DUNG")
    except Exception as e:
        print("\nLOI: %s" % str(e))
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # Kiểm tra dependencies
    required_packages = ['numpy', 'matplotlib']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print("THIEU PACKAGE: %s" % package)
            print("   Cai dat: pip install %s" % package)
            sys.exit(1)
    
    # Chạy chương trình
    main()