"""
RUN_ALL_WITH_PLOTS.PY - File chính để chạy tất cả chức năng
"""

import numpy as np
import matplotlib
import sys
import os
import subprocess

# Import các module gốc
sys.path.append('.')

def run_simulation():
    """Chạy simulation chính và trả về dữ liệu"""
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

def create_video_directly():
    """Tạo video trực tiếp bằng cách chạy file video_animation.py"""
    print("\n" + "="*70)
    print("BAT DAU TAO VIDEO")
    print("="*70)
    
    try:
        # Kiểm tra file video_animation.py tồn tại
        if not os.path.exists('video_animation.py'):
            print("LOI: Khong tim thay file video_animation.py")
            return False
        
        print("Dang chay video_animation.py...")
        
        # Chạy file video_animation.py như một chương trình độc lập
        result = subprocess.run([sys.executable, 'video_animation.py'], 
                              capture_output=True, text=True)
        
        # In kết quả
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("LOI:", result.stderr)
        
        # Kiểm tra xem video đã được tạo chưa
        video_files = ['drone_formation_video.mp4', 'drone_formation_video.gif']
        for video_file in video_files:
            if os.path.exists(video_file):
                print(f"\nTHANH CONG: Da tao video: {video_file}")
                return True
        
        # Kiểm tra thư mục animation_frames
        if os.path.exists('animation_frames'):
            print("\nTHANH CONG: Da luu cac frame trong thu muc 'animation_frames/'")
            return True
        
        print("\nCANH BAO: Khong tim thay file video nao duoc tao")
        return False
        
    except Exception as e:
        print(f"LOI khi tao video: {e}")
        return False

def main():
    """Hàm chính - chạy tất cả"""
    print("="*80)
    print("DO AN TOT NGHIEP: DIEU KHIEN DOI HINH HINH CHU V CHO NHIEU UAV")
    print("="*80)
    
    try:
        # Tạo thư mục output
        os.makedirs('output', exist_ok=True)
        
        # 1. Chạy simulation chính
        drones, model, iterations = run_simulation()
        
        # 2. Import và gọi hàm vẽ đồ thị
        print("\n" + "="*70)
        print("VE DO THI DANH GIA")
        print("="*70)
        
        try:
            from plot_figures import calculate_metrics, plot_trajectory, plot_figure_6, print_statistics
            
            # Vẽ đường đi
            plot_trajectory(drones, model)
            
            # Tính metrics
            metrics = calculate_metrics(drones, model)
            
            # Vẽ 4 đồ thị như bài báo
            plot_figure_6(metrics, model)
            
            # In thống kê
            print_statistics(metrics, model)
            
        except ImportError as e:
            print(f"Khong the import plot_figures: {e}")
            print("Dam bao file plot_figures.py co trong thu muc")
        
        # 3. Hỏi người dùng có muốn tạo video không
        print("\n" + "="*70)
        print("TAO VIDEO ANIMATION")
        print("="*70)
        
        response = input("\nBan co muon tao video animation? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            # Tạo video bằng cách chạy file video_animation.py độc lập
            video_created = create_video_directly()
            
            if not video_created:
                print("\nThu tao video bang cach khac...")
                try:
                    # Thử import và chạy trực tiếp
                    import video_animation
                    video_animation.main()
                except Exception as e:
                    print(f"Loi khi tao video: {e}")
                    print("Co the can cai dat ffmpeg de tao video MP4")
                    print("Hoac video se duoc luu duoi dang GIF hoac cac frame rieng le")
        
        # Hiển thị tổng kết
        print("\n" + "="*70)
        print("TONG KET KET QUA")
        print("="*70)
        
        print("\nCac file da tao:")
        print("  Trong thu muc 'output/':")
        
        output_files = os.listdir('output') if os.path.exists('output') else []
        for file in output_files:
            if file.endswith('.png'):
                print(f"    - {file}")
        
        print("\n  Trong thu muc goc:")
        for file in os.listdir('.'):
            if file.endswith('.mp4') or file.endswith('.gif'):
                print(f"    - {file}")
        
        if os.path.exists('animation_frames'):
            frame_count = len([f for f in os.listdir('animation_frames') if f.endswith('.png')])
            print(f"    - animation_frames/ ({frame_count} frames)")
        
        print("\nThoi gian simulation: %.2f giay" % (iterations * 0.02))
        print("So UAV: %d" % model['n'])
        print("Khoang cach mong muon: %.2f m" % model['d'])
        print("Goc doi hinh: %.2f rad (%.1f do)" % (model['alpha'], model['alpha'] * 180 / np.pi))
        
    except KeyboardInterrupt:
        print("\n\nDA DUNG BOI NGUOI DUNG")
    except Exception as e:
        print("\nLOI: %s" % str(e))
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("KET THUC CHUONG TRINH")
    print("="*80)

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