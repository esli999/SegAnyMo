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
import shutil


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


def process_single_video(video_path, output_dir, processing_dir, gpu, efficiency, config, step, script_dir):
    """Process a single video through the complete pipeline."""
    
    # Validate video exists
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return False, 0
    
    # Get absolute paths
    video_path = os.path.abspath(video_path)
    output_dir = os.path.abspath(output_dir)
    processing_dir = os.path.abspath(processing_dir)
    config_path = os.path.join(script_dir, config)
    
    # Create processing directory and copy video there
    os.makedirs(processing_dir, exist_ok=True)
    video_name = Path(video_path).stem
    video_ext = Path(video_path).suffix
    processing_video_path = os.path.join(processing_dir, f"{video_name}{video_ext}")
    
    # Copy video to processing directory if not already there
    if not os.path.exists(processing_video_path):
        print(f"Copying video to processing directory: {processing_dir}")
        shutil.copy2(video_path, processing_video_path)
    
    # Create output directories
    moseg_dir = os.path.join(output_dir, 'moseg')
    sam2_dir = os.path.join(output_dir, 'sam2')
    os.makedirs(moseg_dir, exist_ok=True)
    os.makedirs(sam2_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"Processing: {video_name}")
    print("="*80)
    print(f"Source Video:        {video_path}")
    print(f"Processing Video:    {processing_video_path}")
    print(f"Processing Directory: {processing_dir}")
    print(f"Output Directory:    {output_dir}")
    print(f"GPU ID:              {gpu}")
    print(f"Efficiency Mode:     {efficiency}")
    print(f"Config File:         {config_path}")
    print(f"Frame Stride:        {step}")
    print("="*80 + "\n")
    
    # Build efficiency flag
    efficiency_flag = "--e" if efficiency else ""
    
    # Change to script directory
    original_dir = os.getcwd()
    os.chdir(script_dir)
    
    start_time = time.time()
    
    try:
        # Step 1: Data Preprocessing (will create intermediate files in processing_dir)
        step1_cmd = (
            f"python core/utils/run_inference.py "
            f"--video_path {processing_video_path} "
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
            f"--video_path {processing_video_path} "
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
            f"--video_path {processing_video_path} "
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
    # Gestalt experiment parameters
    SCENES = [f'scene_{i:05d}' for i in range(20)]  # scene_00000 to scene_00019
    TEXTURES = ['texture_00', 'texture_07', 'texture_13', 'texture_16', 'texture_21', 'texture_22', 'texture_25']
    BASE_PATH = '/home/esli/GenParticles_neural_stimulus/assets/from_thomas'
    VIDEO_FILENAME = 'output_six_frame.mp4'
    
    parser = argparse.ArgumentParser(
        description="Process Gestalt videos through the SegAnyMo pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all gestalt videos with default settings
  python process_gestalt_videos.py
  
  # With efficiency mode
  python process_gestalt_videos.py --efficiency
  
  # Use specific GPU
  python process_gestalt_videos.py --gpu 1
        """
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='gestalt_SegAnyMo_outputs',
        help='Directory to save all outputs (default: gestalt_SegAnyMo_outputs)'
    )
    
    parser.add_argument(
        '--processing_dir',
        type=str,
        default='SegAnyMo_processing_files',
        help='Directory to store intermediate processing files (default: SegAnyMo_processing_files)'
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
        '--continue_on_error',
        action='store_true',
        help='Continue processing remaining videos if one fails'
    )
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate all video paths from scene/texture combinations
    video_files = []
    missing_videos = []
    
    for scene in SCENES:
        for texture in TEXTURES:
            video_path = os.path.join(BASE_PATH, scene, texture, VIDEO_FILENAME)
            if os.path.exists(video_path):
                video_files.append(video_path)
            else:
                missing_videos.append(video_path)
    
    # Report on found vs missing videos
    print(f"\nFound {len(video_files)} videos out of {len(SCENES) * len(TEXTURES)} expected")
    if missing_videos:
        print(f"Missing {len(missing_videos)} videos:")
        for missing in missing_videos[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_videos) > 10:
            print(f"  ... and {len(missing_videos) - 10} more")
    
    if not video_files:
        print("[ERROR] No videos found to process!")
        sys.exit(1)
    
    batch_mode = True
    
    # Print processing summary
    print("\n" + "="*80)
    print("SegAnyMo Gestalt Video Processing Pipeline")
    print("="*80)
    print(f"Scenes:                {len(SCENES)} (scene_00000 to scene_{len(SCENES)-1:05d})")
    print(f"Textures:              {len(TEXTURES)} ({', '.join(TEXTURES)})")
    print(f"Videos to process:     {len(video_files)}")
    print(f"Processing directory:  {args.processing_dir}")
    print(f"Output directory:      {args.output_dir}")
    print(f"GPU ID:                {args.gpu}")
    print(f"Efficiency mode:       {args.efficiency}")
    print("="*80)
    
    # Process videos
    results = []
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n\n{'#'*80}")
        
        # Extract scene and texture from path
        path_parts = Path(video_path).parts
        scene = path_parts[-3]  # e.g., 'scene_00000'
        texture = path_parts[-2]  # e.g., 'texture_00'
        
        print(f"# Video {i}/{len(video_files)}: {scene}/{texture}")
        print(f"{'#'*80}\n")
        
        # Create subdirectory name: scene_texture
        subdir_name = f"{scene}_{texture}"
        video_output_dir = os.path.join(args.output_dir, subdir_name)
        
        # Create processing directory that mirrors the scene/texture structure
        video_processing_dir = os.path.join(args.processing_dir, scene, texture)
        
        success, elapsed = process_single_video(
            video_path=video_path,
            output_dir=video_output_dir,
            processing_dir=video_processing_dir,
            gpu=args.gpu,
            efficiency=args.efficiency,
            config=args.config,
            step=args.step,
            script_dir=script_dir
        )
        
        results.append({
            'video': f"{scene}/{texture}",
            'success': success,
            'time': elapsed
        })
        
        if not success and not args.continue_on_error:
            print("\n[ERROR] Processing failed. Stopping batch processing.")
            print("Use --continue_on_error to continue despite failures.")
            break
    
    # Print final summary
    total_elapsed = time.time() - total_start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print("\n\n" + "="*80)
    print("GESTALT VIDEO PROCESSING COMPLETE")
    print("="*80)
    print(f"\nTotal time:        {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"                   ({total_elapsed/3600:.2f} hours)")
    print(f"Videos processed:  {len(results)}")
    print(f"Successful:        {successful} ✅")
    print(f"Failed:            {failed} ❌")
    print(f"\nAverage time per video: {total_elapsed/len(results):.1f} seconds")
    print("\nDetailed Results:")
    print("-" * 80)
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        time_str = f"{result['time']:.1f}s"
        print(f"  {status:12} | {time_str:10} | {result['video']}")
    
    print("="*80)
    print(f"\nFinal outputs saved to:       {args.output_dir}")
    print(f"Intermediate files saved to:  {args.processing_dir}")
    print("="*80 + "\n")
    
    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

