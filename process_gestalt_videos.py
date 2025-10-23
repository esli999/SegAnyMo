#!/usr/bin/env python
"""
SegAnyMo Video Processing Pipeline
Processes single or multiple videos through the complete SegAnyMo pipeline:
1. Data preprocessing (depths, tracks, dinos)
2. Motion segmentation inference
3. Final mask generation with SAM2

Usage:
    # Process a single video
    python process_videos.py --video path/to/video.mp4 --output_dir ./results
    
    # Process all videos in a directory
    python process_videos.py --input_dir ./videos --output_dir ./results
    
    # Process specific videos
    python process_videos.py --videos video1.mp4 video2.mp4 video3.mp4 --output_dir ./results
    
    # With options
    python process_videos.py --video video.mp4 --output_dir ./results --gpu 0 --efficiency
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import time


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(command, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Failed during: {description}")
        return False
    
    print(f"\n[SUCCESS] Completed: {description}\n")
    return True


def process_single_video(video_path, output_dir, gpu, efficiency, config, step, script_dir):
    """Process a single video through the complete pipeline."""
    
    # Validate video exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False, 0
    
    # Get absolute paths
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    config_path = os.path.join(script_dir, config)
    
    # Create output directories
    moseg_dir = os.path.join(output_dir, 'moseg')
    sam2_dir = os.path.join(output_dir, 'sam2')
    os.makedirs(moseg_dir, exist_ok=True)
    os.makedirs(sam2_dir, exist_ok=True)
    
    # Get video name
    video_name = Path(video_path).stem
    
    print("\n" + "="*80)
    print(f"Processing: {video_name}")
    print("="*80)
    print(f"Input Video:      {video_path}")
    print(f"Output Directory: {output_dir}")
    print(f"GPU ID:           {gpu}")
    print(f"Efficiency Mode:  {efficiency}")
    print(f"Config File:      {config_path}")
    print(f"Frame Stride:     {step}")
    print("="*80 + "\n")
    
    # Build efficiency flag
    efficiency_flag = "--e" if efficiency else ""
    
    # Change to script directory
    original_dir = os.getcwd()
    os.chdir(script_dir)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Preprocessing
        step1_cmd = (
            f"python core/utils/run_inference.py "
            f"--video_path {video_path} "
            f"--gpus {gpu} "
            f"--depths --tracks --dinos "
            f"--step {step} "
            f"{efficiency_flag}"
        )
        
        if not run_command(step1_cmd, "Step 1: Data Preprocessing (depths, tracks, dinos)"):
            return False, time.time() - start_time
        
        # Step 2: Motion Segmentation Inference
        step2_cmd = (
            f"python core/utils/run_inference.py "
            f"--video_path {video_path} "
            f"--motin_seg_dir {moseg_dir} "
            f"--config_file {config_path} "
            f"--gpus {gpu} "
            f"--motion_seg_infer "
            f"--step {step} "
            f"{efficiency_flag}"
        )
        
        if not run_command(step2_cmd, "Step 2: Motion Segmentation Inference"):
            return False, time.time() - start_time
        
        # Step 3: Final Mask Generation with SAM2
        step3_cmd = (
            f"python core/utils/run_inference.py "
            f"--video_path {video_path} "
            f"--sam2dir {sam2_dir} "
            f"--motin_seg_dir {moseg_dir} "
            f"--gpus {gpu} "
            f"--sam2 "
            f"--step {step} "
            f"{efficiency_flag}"
        )
        
        if not run_command(step3_cmd, "Step 3: Final Mask Generation with SAM2"):
            return False, time.time() - start_time
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*80)
        print(f"✅ COMPLETED: {video_name}")
        print("="*80)
        print(f"Processing time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"\n📁 Results saved to: {output_dir}")
        print(f"\n   Output Structure:")
        print(f"   {output_dir}/")
        print(f"   ├── moseg/")
        print(f"   │   └── {video_name}/")
        print(f"   │       ├── original.mp4")
        print(f"   │       ├── dynamic.mp4")
        print(f"   │       ├── dynamic_traj.npy")
        print(f"   │       ├── dynamic_confidences.npy")
        print(f"   │       └── dynamic_visibility.npy")
        print(f"   └── sam2/")
        print(f"       ├── initial_preds/{video_name}/*.png")
        print(f"       └── final_res/{video_name}/video/")
        print(f"           ├── mask.mp4")
        print(f"           ├── mask_rgb.mp4")
        print(f"           ├── mask_rgb_color.mp4  ⭐ (main output)")
        print(f"           └── original_rgb.mp4")
        print("="*80 + "\n")
        
        return True, elapsed_time
        
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Process video(s) through the SegAnyMo pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python process_videos.py --video video.mp4 --output_dir ./results
  
  # All videos in directory
  python process_videos.py --input_dir ./videos --output_dir ./results
  
  # Specific videos
  python process_videos.py --videos v1.mp4 v2.mp4 --output_dir ./results
  
  # With efficiency mode
  python process_videos.py --video video.mp4 --output_dir ./results --efficiency
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--video',
        type=str,
        help='Single video file to process'
    )
    input_group.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing videos to process'
    )
    input_group.add_argument(
        '--videos',
        nargs='+',
        help='List of specific video files to process'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save outputs (subdirectory per video in batch mode)'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU ID to use (default: 0)'
    )
    
    parser.add_argument(
        '--efficiency',
        action='store_true',
        help='Enable efficiency mode (frame rate reduction, resolution scaling)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/example_train.yaml',
        help='Path to config file (default: ./configs/example_train.yaml)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        default=10,
        help='Frame stride for processing (default: 10)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.mp4',
        help='File pattern for input_dir mode (default: *.mp4)'
    )
    
    parser.add_argument(
        '--continue_on_error',
        action='store_true',
        help='Continue processing remaining videos if one fails (batch mode only)'
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Collect video files
    video_files = []
    batch_mode = False
    
    if args.video:
        # Single video mode
        video_files = [args.video]
        
    elif args.input_dir:
        # Directory mode
        batch_mode = True
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] Input directory not found: {args.input_dir}")
            sys.exit(1)
        
        video_files = sorted(input_dir.glob(args.pattern))
        video_files = [str(f) for f in video_files]
        
        if not video_files:
            print(f"[ERROR] No videos found matching pattern '{args.pattern}' in {args.input_dir}")
            sys.exit(1)
    
    elif args.videos:
        # Multiple videos mode
        batch_mode = True
        for video in args.videos:
            if not os.path.exists(video):
                print(f"[WARNING] Video not found, skipping: {video}")
            else:
                video_files.append(video)
        
        if not video_files:
            print("[ERROR] None of the specified videos exist")
            sys.exit(1)
    
    # Print processing summary
    print("\n" + "="*80)
    print("SegAnyMo Video Processing Pipeline")
    print("="*80)
    print(f"Mode:             {'BATCH' if batch_mode else 'SINGLE'}")
    print(f"Videos to process: {len(video_files)}")
    print(f"Output directory:  {args.output_dir}")
    print(f"GPU ID:            {args.gpu}")
    print(f"Efficiency mode:   {args.efficiency}")
    print("="*80)
    
    if len(video_files) > 1:
        print("\nVideos:")
        for i, video in enumerate(video_files, 1):
            print(f"  {i}. {Path(video).name}")
        print("="*80)
    
    # Process videos
    results = []
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        if batch_mode:
            print(f"\n\n{'#'*80}")
            print(f"# Video {i}/{len(video_files)}: {Path(video_path).name}")
            print(f"{'#'*80}\n")
            
            # Each video gets its own subdirectory in batch mode
            video_name = Path(video_path).stem
            video_output_dir = os.path.join(args.output_dir, video_name)
        else:
            # Single video uses the output_dir directly
            video_output_dir = args.output_dir
        
        success, elapsed = process_single_video(
            video_path=video_path,
            output_dir=video_output_dir,
            gpu=args.gpu,
            efficiency=args.efficiency,
            config=args.config,
            step=args.step,
            script_dir=script_dir
        )
        
        results.append({
            'video': Path(video_path).name,
            'success': success,
            'time': elapsed
        })
        
        if not success and not args.continue_on_error:
            if batch_mode:
                print("\n[ERROR] Processing failed. Stopping batch processing.")
                print("Use --continue_on_error to continue despite failures.")
            break
    
    # Print final summary for batch mode
    if batch_mode:
        total_elapsed = time.time() - total_start_time
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print("\n\n" + "="*80)
        print("BATCH PROCESSING COMPLETE")
        print("="*80)
        print(f"\nTotal time:        {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
        print(f"Videos processed:  {len(results)}")
        print(f"Successful:        {successful} ✅")
        print(f"Failed:            {failed} ❌")
        print("\nDetailed Results:")
        print("-" * 80)
        
        for result in results:
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            time_str = f"{result['time']:.1f}s"
            print(f"  {status:12} | {time_str:10} | {result['video']}")
        
        print("="*80)
        print(f"\nAll outputs saved to: {args.output_dir}")
        print("="*80 + "\n")
        
        # Exit with error code if any failed
        if failed > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()

