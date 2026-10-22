import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from videostream import VideoStream
from Model import Model
from crop import extract_and_crop_object, create_gamma_lut


def build_background(stream: VideoStream, resize_ratio: float = 0.25, num_frames: int = 20, blur_ksize: tuple = (17, 17), debug: bool = False) -> np.ndarray:
    """
    Reads the initial `num_frames` from the stream, averages them into a mean background,
    and returns the blurred background.
    """
    print(f"Collecting {num_frames} initial frames to compute background model...")
    bg_frames = []
    
    while len(bg_frames) < num_frames and stream.is_alive():
        frame = stream.get_frame()
        if frame is not None:
            frame = cv2.resize(frame, None, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
            bg_frames.append(frame)
        if not debug:
            time.sleep(1.0 / stream.get_fps())

    if not bg_frames:
        raise RuntimeError("Failed to collect background frames from video stream.")

    bg_mean = np.mean(np.stack(bg_frames, axis=0), axis=0).astype(np.uint8)
    bg_blurred = cv2.blur(bg_mean, blur_ksize)
    print("Background model created successfully.")
    return bg_blurred


def draw_overlay(
    frame: np.ndarray,
    resize_ratio: float,
    main_counter: Counter,
    temp_counter: Counter,
    has_active_object: bool,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    current_label: Optional[str] = None,
) -> np.ndarray:
    """Draws bounding box on the object, main counter on top, and temp counter at bottom."""
    vis_frame = frame.copy()
    h, w = vis_frame.shape[:2]

    # --- Draw Bounding Box around active object ---
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1 / resize_ratio), int(y1 / resize_ratio), int(x2 / resize_ratio), int(y2 / resize_ratio)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if current_label:
            label_text = f"{current_label}"
            # Text background tag
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(
                vis_frame, label_text, (x1 + 3, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

    # --- Top Banner: Main Counter ---
    active_main = [f"{cls}: {cnt}" for cls, cnt in main_counter.items() if cnt > 0]
    main_text = "TOTAL: " + (" | ".join(active_main) if active_main else "0 items")

    # Dark background banner for top text
    cv2.rectangle(vis_frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(
        vis_frame, main_text, (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA
    )

    # --- Bottom Banner: Temp Counter or 'No object' ---
    cv2.rectangle(vis_frame, (0, h - 45), (w, h), (0, 0, 0), -1)

    if has_active_object and sum(temp_counter.values()) > 0:
        top_cls, top_cnt = temp_counter.most_common(1)[0]
        temp_text = f"TRACKING: {top_cls} (votes: {top_cnt}) | Total detections: {sum(temp_counter.values())}"
        text_color = (0, 255, 0)  # Green
    else:
        temp_text = "STATUS: No object"
        text_color = (180, 180, 180)  # Gray

    cv2.putText(
        vis_frame, temp_text, (15, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA
    )

    return vis_frame


def main(debug: bool = True):
    video_path = "/home/filelele/personal_project/belt-conveyor-object-count/inference/input/video_test_1.avi"
    checkpoint_path = "/home/filelele/personal_project/belt-conveyor-object-count/training/checkpoints/mobileone_s0_final_fused.pth"
    output_video_path = "/home/filelele/personal_project/belt-conveyor-object-count/inference/output/output_counted.mp4"
    
    crop_times_no_object = []
    crop_times_with_object = []
    classification_times = []
    e2e_times = []
    
    # Ensure parent output directory exists
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Video Stream (pass debug mode) & Model
    stream = VideoStream(video_path, debug=debug)
    model = Model(checkpoint_path=checkpoint_path, variant="s0")

    # Resize ratio
    resize_ratio = 0.25

    # 2. Build background model from the first 20 frames
    bg_blurred = build_background(stream, resize_ratio=resize_ratio,num_frames=20, debug=debug)
    lut_table = create_gamma_lut(gamma=1.5)

    # 3. Setup VideoWriter
    fps = stream.get_fps()
    first_frame = stream.get_frame()
    if first_frame is None:
        raise RuntimeError("Could not retrieve frame to initialize VideoWriter.")
    frame_h, frame_w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))

    # 4. Counters
    main_counter = Counter({c: 0 for c in model.class_names})
    temp_counter = Counter({c: 0 for c in model.class_names})
    none_consecutive = 0
    has_active_object = False

    MIN_NONE_COUNT = 60      # Minimum consecutive None frames to consider the object left
    MIN_DETECTION_COUNT = 3 # Minimum appearances in temp_counter to count as valid item

    print("\nStarting conveyor object counting loop...")
    print(f"Stream FPS: {fps:.2f} | Debug Mode (No Wait): {debug}")
    print(f"Saving output video to: {output_video_path}")

    while stream.is_alive():
        start_time = time.perf_counter()
        frame = stream.get_frame()
        if frame is None:
            break
        resized_frame = cv2.resize(frame, None, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)

        crop_start_time = time.perf_counter()
        # Extract and crop object if fully in view
        cropped_obj, bbox = extract_and_crop_object(
            resized_frame,    
            bg_blurred=bg_blurred, 
            lut_table=lut_table,
            blur_ksize=(17,17),
            threshold_value=150,
            close_ksize=51,
            padding_pct=0.05,
            edge_margin=0
        )
        crop_end_time = time.perf_counter()
        if bbox is not None:
            crop_times_with_object.append(crop_end_time - crop_start_time)
        else:
            crop_times_no_object.append(crop_end_time - crop_start_time)
        current_label = None
        if cropped_obj is not None:
            # Object is in view
            none_consecutive = 0
            has_active_object = True
            current_label, inference_time = model.process_frame(cropped_obj)
            temp_counter[current_label] += 1
            classification_times.append(inference_time)
        else:
            # No object detected or object touching frame edge
            none_consecutive += 1

            if none_consecutive >= MIN_NONE_COUNT:
                # Trigger state change: item has passed
                if has_active_object and temp_counter:
                    top_class, top_count = temp_counter.most_common(1)[0]
                    if top_count >= MIN_DETECTION_COUNT:
                        main_counter[top_class] += 1
                        print(f"\n[COUNTED] Item: {top_class} (Total {top_class}: {main_counter[top_class]})")
                        print(f"Current Main Counter: {dict(main_counter)}")

                    # Reset temp counter and hide tracking text
                    temp_counter.clear()

                has_active_object = False
        e2e_times.append(time.perf_counter() - start_time)
        # Render overlays (with bounding box) and write to output video
        annotated_frame = draw_overlay(
            frame, 
            resize_ratio,
            main_counter, 
            temp_counter, 
            has_active_object,
            bbox=bbox,
            current_label=current_label
        )
        out_writer.write(annotated_frame)

        # Match stream pace (bypassed in debug mode)
        if not debug:
            elapsed = time.perf_counter() - start_time
            sleep_time = (1.0 / fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Release resources
    stream.stop()
    out_writer.release()
    print(f"\nOutput video saved successfully -> {output_video_path}")

    print("\n" + "=" * 40)
    print("Inference finished. Final Counts:")
    for cls, count in main_counter.items():
        if count > 0:
            print(f"  {cls:<15}: {count}")
    print("=" * 40)
    print("Pipeline latency:")
    print(f"Crop time: {sum(crop_times_no_object[-100:])/len(crop_times_no_object[-100:])*1000:.4f} ms")
    print(f"Crop time (with object): {sum(crop_times_with_object[-100:])/len(crop_times_with_object[-100:])*1000:.4f} ms")
    print(f"Classification time: {sum(classification_times[-100:])/len(classification_times[-100:])*1000:.4f} ms")
    print(f"E2E time: {sum(e2e_times[-100:])/len(e2e_times[-100:])*1000:.4f} ms")

if __name__ == "__main__":
    main()
