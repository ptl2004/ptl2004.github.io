"""
RUN_ALL.PY - Run all simulations
"""
import subprocess
import sys

def run_simulation():
    """Run the main simulation"""
    print("="*60)
    print("RUNNING MAIN SIMULATION")
    print("="*60)
    
    try:
        import main
        main.main()
        return True
    except Exception as e:
        print(f"Error in main simulation: {e}")
        return False

def run_video():
    """Run video creation"""
    print("\n" + "="*60)
    print("CREATING ANIMATION VIDEO")
    print("="*60)
    
    try:
        import video_animation
        video_animation.main()
        return True
    except ImportError as e:
        print(f"Video creation requires additional packages.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return False
    except Exception as e:
        print(f"Error in video creation: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required = ['numpy', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def main():
    """Main function to run everything"""
    print("="*60)
    print("DRONE FORMATION SIMULATION - COMPLETE PACKAGE")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\nWhat would you like to run?")
    print("1. Main simulation (static plot)")
    print("2. Animation video")
    print("3. Both")
    print("4. Test individual components")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice in ['1', '3']:
        run_simulation()
    
    if choice in ['2', '3']:
        run_video()
    
    if choice == '4':
        print("\nTesting components...")
        
        # Test CreateModel2
        try:
            from CreateModel2 import CreateModel2
            model = CreateModel2()
            print("✓ CreateModel2: OK")
        except Exception as e:
            print(f"✗ CreateModel2: {e}")
        
        # Test Drone
        try:
            from Drone import Drone
            drone = Drone(1, [0, 0])
            print("✓ Drone: OK")
        except Exception as e:
            print(f"✗ Drone: {e}")
        
        # Test Perpendicular
        try:
            from Perpendicular import Perpendicular
            result = Perpendicular([0,0], [-1,1], [1,1])
            print(f"✓ Perpendicular: OK (result: {result})")
        except Exception as e:
            print(f"✗ Perpendicular: {e}")
    
    print("\n" + "="*60)
    print("PROCESS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()