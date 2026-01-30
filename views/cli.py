from controllers.video_processing import process_video
import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="A script to process video files and output results."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input video file (e.g., video.mp4)",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output video (e.g., out.mp4)",
    )

    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to save the metadata JSON (e.g., out.json)",
    )

    args = parser.parse_args()

    video_path = args.input
    output_video_path = args.output
    json_output_path = args.json
    try:
        vido_annotation = process_video(video_path, output_video_path)
    except Exception as e:
        raise RuntimeError(f"Failed to process the video due to: {e}")
    try:
        with open(json_output_path, "w", encoding="utf-8") as f:
            # Convert model to a dict first, then dump to file
            json.dump(vido_annotation.model_dump(), f, indent=4)

        print(f"Successfully saved metadata to: {args.json}")
    except IOError as e:
        print(f"Error: Could not write to file {args.json}. {e}")
