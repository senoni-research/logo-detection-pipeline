#!/usr/bin/env python3
"""
Logo Detection Pipeline Setup and Runner
========================================

This script provides an easy interface to set up and run the logo detection pipeline.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_system_requirements():
    """Check if system meets requirements"""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check for CUDA (optional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available, using CPU (slower but functional)")
    except ImportError:
        print("ℹ️  PyTorch not yet installed")
    
    return True

def create_sample_data():
    """Create sample data structure"""
    dirs = ["videos", "templates", "results"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create a simple README for each directory
    readme_content = {
        "videos/README.md": "Place your video files (.mp4, .avi, .mov) here for analysis.",
        "templates/README.md": "Place your logo template images (.png, .jpg) here.",
        "results/README.md": "Detection results and reports will be saved here."
    }
    
    for file_path, content in readme_content.items():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write(content)
    
    print("✅ Directory structure created")

def run_demo():
    """Run the demo notebook"""
    print("Starting Jupyter notebook demo...")
    try:
        subprocess.run([sys.executable, "-m", "jupyter", "notebook", "logo_detection_demo.ipynb"])
    except KeyboardInterrupt:
        print("Demo stopped by user")
    except Exception as e:
        print(f"Error running demo: {e}")

def run_pipeline_cli(args):
    """Run pipeline via command line"""
    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.template):
        print(f"❌ Template file not found: {args.template}")
        return
    
    print(f"Running logo detection for {args.logo_name}...")
    
    # Import and run pipeline
    try:
        from logo_detection_pipeline import LogoDetectionPipeline
        
        pipeline = LogoDetectionPipeline(
            use_zero_shot=args.use_zero_shot,
            confidence_threshold=args.confidence,
            template_match_threshold=args.template_threshold
        )
        
        results = pipeline.detect_logo_in_video(
            video_path=args.video,
            template_path=args.template,
            logo_name=args.logo_name,
            output_dir=args.output_dir
        )
        
        print(f"\n🎉 Detection complete!")
        print(f"📊 Results: {results['total_detections']} detections in {results['processing_time']:.1f}s")
        print(f"📁 Saved to: {args.output_dir}")
        
    except ImportError as e:
        print(f"❌ Error importing pipeline: {e}")
        print("Make sure all requirements are installed.")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Logo Detection Pipeline Setup and Runner")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Install requirements and setup environment")
    setup_parser.add_argument("--skip-install", action="store_true", help="Skip package installation")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run Jupyter notebook demo")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run logo detection pipeline")
    run_parser.add_argument("--video", required=True, help="Path to video file")
    run_parser.add_argument("--template", required=True, help="Path to logo template image")
    run_parser.add_argument("--logo_name", required=True, help="Name of the logo")
    run_parser.add_argument("--output_dir", default="results", help="Output directory")
    run_parser.add_argument("--use_zero_shot", action="store_true", help="Enable zero-shot detection")
    run_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    run_parser.add_argument("--template_threshold", type=float, default=0.3, help="Template matching threshold")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        print("🚀 Setting up Logo Detection Pipeline...")
        
        if not check_system_requirements():
            sys.exit(1)
        
        if not args.skip_install:
            if not install_requirements():
                sys.exit(1)
        
        create_sample_data()
        print("\n✅ Setup complete!")
        print("\nNext steps:")
        print("1. Place your video files in the 'videos/' directory")
        print("2. Place your logo templates in the 'templates/' directory")
        print("3. Run: python setup_and_run.py demo")
        print("   OR: python setup_and_run.py run --video videos/your_video.mp4 --template templates/logo.png --logo_name YourBrand")
    
    elif args.command == "demo":
        run_demo()
    
    elif args.command == "run":
        run_pipeline_cli(args)
    
    else:
        print("Logo Detection Pipeline")
        print("======================")
        print()
        print("Available commands:")
        print("  setup  - Install requirements and setup environment")
        print("  demo   - Run interactive Jupyter notebook demo")
        print("  run    - Run logo detection on specific video")
        print()
        print("Examples:")
        print("  python setup_and_run.py setup")
        print("  python setup_and_run.py demo")
        print("  python setup_and_run.py run --video sample.mp4 --template nike.png --logo_name Nike --use_zero_shot")
        print()
        print("For detailed help: python setup_and_run.py <command> --help")

if __name__ == "__main__":
    main() 