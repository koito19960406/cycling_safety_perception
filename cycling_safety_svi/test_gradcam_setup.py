#!/usr/bin/env python3
"""
Test script to verify Grad-CAM setup and dependencies.
Run this before attempting full Grad-CAM visualization.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('matplotlib.pyplot', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('tqdm', 'tqdm')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} - OK")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install torch torchvision pandas numpy opencv-python matplotlib pillow tqdm")
        return False
    
    print("All imports successful!")
    return True


def test_submodule():
    """Test if the cycling safety submodule can be imported."""
    print("\nTesting submodule access...")
    
    current_dir = 'cycling_safety_svi/cycling_safety_subjective_learning_pairwise'
    if not os.path.exists(current_dir):
        print(f"✗ Submodule directory not found: {current_dir}")
        return False
    
    sys.path.append(current_dir)
    
    try:
        from nets.cnn import CNN
        print("✓ CNN class import - OK")
        return True
    except ImportError as e:
        print(f"✗ CNN class import failed: {e}")
        return False


def test_model_file():
    """Test if the default model file exists."""
    print("\nTesting model file...")
    
    model_path = 'cycling_safety_svi/cycling_safety_subjective_learning_pairwise/models/vgg_syn+ber.pt'
    
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
        return True
    else:
        print(f"✗ Model file not found: {model_path}")
        print("Please ensure you have the trained model file in the models directory.")
        return False


def test_gpu():
    """Test GPU availability."""
    print("\nTesting GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available - {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("! CUDA not available - will use CPU")
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")


def test_output_directory():
    """Test if output directory can be created."""
    print("\nTesting output directory...")
    
    output_dir = '/home/kiito/cycling_safety_svi/data/processed'
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, 'test_write.txt')
        
        with open(test_file, 'w') as f:
            f.write('test')
        
        os.remove(test_file)
        print(f"✓ Output directory writable: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Cannot write to output directory {output_dir}: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Grad-CAM Setup Test ===\n")
    
    tests = [
        test_imports,
        test_submodule,
        test_model_file,
        test_gpu,
        test_output_directory
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            results.append(False)
    
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("Setup is ready for Grad-CAM visualization!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed ({passed}/{total} passed)")
        print("Please fix the issues above before running Grad-CAM visualization.")
        return 1


if __name__ == '__main__':
    exit(main())