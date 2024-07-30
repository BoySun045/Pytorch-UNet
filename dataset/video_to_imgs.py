import cv2
import os
import argparse
from tqdm import tqdm

def sample_images_from_video(video_path, output_folder, sample_interval):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0
    
    for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_idx % sample_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{frame_idx}.jpg")
            cv2.imwrite(image_path, frame)
            saved_count += 1
    
    cap.release()
    print(f"Saved {saved_count} images to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Sample images from a video at specified intervals.")
    parser.add_argument('--video_path', '-in' ,type=str, help="Path to the input video file.")
    parser.add_argument('--output_folder', '-out', type=str, help="Folder to save the sampled images.")
    parser.add_argument('--sample_interval', '-inter', type=int, help="Interval (in frames) between each sampled image.")
    
    args = parser.parse_args()
    
    sample_images_from_video(args.video_path, args.output_folder, args.sample_interval)

if __name__ == "__main__":
    main()
