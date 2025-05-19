import argparse
import numpy as np
import cv2

from ocsort.trackers.ocsort_tracker.ocsort import OCSort

def generate_non_linear_bbox(frame_idx, start_pos=(750, 100), velocity=(-2, 1.5), size=(60, 40)):
  """Generate a bounding box that moves in a non-linear path."""
  x = start_pos[0] + velocity[0] * frame_idx
  y = start_pos[1] + int(50 * np.sin(frame_idx / 5.0))  # Non-linear motion
  w, h = size
  return np.array([x, y, x + w, y + h, 0.99])


def generate_linear_bbox(frame_idx, start_pos=(750, 100), velocity=(-2, 1.5), size=(60, 40)):
  """Generate a bounding box that moves linearly from right to left."""
  x = start_pos[0] + velocity[0] * frame_idx
  y = start_pos[1]
  w, h = size
  return np.array([x, y, x + w, y + h, 0.99])


def generate_accelerating_bbox(frame_idx, start_pos=(750, 100), velocity=(-1, 1.5), size=(60, 40), max_speed=10):
  """Generate a bounding box that accelerates from right to left."""
  x = start_pos[0] + velocity[0] * int(0.3 * frame_idx**2)
  x = min(x, start_pos[0] - max_speed)  # Limit the maximum speed

  y = start_pos[1]
  w, h = size
  return np.array([x, y, x + w, y + h, 0.99])


def visualize_frame_cv2(idx, detections, tracks, frame_size=(600, 800)):
  frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)

  cv2.putText(frame, f'Frame: {idx:3d}', (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  for track in tracks:
    tx1, ty1, tx2, ty2 = track["bbox"]
    track_id = track["track_id"]
    state = track["state"]

    color = (0, 0, 255)  # default red for confirmed

    if state == "tentative":
      color = (0, 255, 255)  # yellow for tentative
    elif state == "confirmed":
      color = (0, 255, 0)  # red

    # Always show tracking state
    cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), color, 2)
    cv2.putText(frame, f'ID: {int(track_id)}', (int(tx1), int(ty1) - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw green circle at center for predicted track
    cx = int((tx1 + tx2) / 2)
    cy = int((ty1 + ty2) / 2)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

  cv2.imshow('OCSort Tracking Simulation', frame)
  return True


def run_interactive_viewer(all_data, frame_size):
  idx = 0
  total = len(all_data)

  while True:
    detections, tracks = all_data[idx]
    draw_success = visualize_frame_cv2(idx, detections, tracks, frame_size)
    if not draw_success:
      break

    key = cv2.waitKey(0)
    if key == 27:  # ESC
      break
    elif key == ord('q'):
      break
    elif key == ord('p'):  # Left arrow
      idx = max(0, idx - 1)
    elif key == ord('n'):  # Right arrow
      idx = min(total - 1, idx + 1)

  cv2.destroyAllWindows()


def main(dropout_frames):
  num_frames = 20 * 20  # 20 seconds at 20 FPS
  frame_size = (600, 800)

  tracker = OCSort(
      det_thresh=0.3,
      iou_threshold=0.55,
      max_age=10,
      min_hits=3,
      delta_t=3,
      asso_func='giou',
      inertia=0.2
  )

  all_data = []

  for frame_idx in range(num_frames):
    detections = []
    if dropout_frames and not ((frame_idx // dropout_frames) % 2 == 1):
      detections.append(generate_linear_bbox(frame_idx, start_pos=(750, 400)))

    detections.append(generate_accelerating_bbox(frame_idx, start_pos=(750, 100), velocity=(-2, 1.5)))
    detections.append(generate_linear_bbox(frame_idx, start_pos=(500, 100)))
    detections.append(generate_linear_bbox(frame_idx, start_pos=(750, 200)))
    detections.append(generate_non_linear_bbox(frame_idx, start_pos=(750, 225), velocity=(-2, 1.5)))

    _ = tracker.update(
        np.array(detections),
        frame_size,
        frame_size
    )

    # Step 2: extract metadata from internal track objects
    visual_tracks = []
    for trk in tracker.trackers:
      # Use your own logic for "confirmed" or "tentative"
      is_confirmed = trk.hits >= tracker.min_hits and trk.time_since_update == 0
      lost_too_long = trk.time_since_update > tracker.max_age
      print(f"Frame {frame_idx:3d} Track ID: {trk.id}, Hits: {trk.hits:3d}, Time since update: {trk.time_since_update:3d}, Age: {trk.age:3d}, Lost: {lost_too_long}")

      if lost_too_long:
        # Skip or remove track if needed
        continue

      bbox = trk.bbox[:4]  # Assuming bbox is in the format [x1, y1, x2, y2]

      if trk.time_since_update > 0:
        bbox = trk.kf.x[:4]  # Use Kalman filter prediction if available

      visual_tracks.append({
          "bbox": trk.history[-1][0] if trk.time_since_update > 0 else bbox,
          "velocity": trk.velocity,
          "track_id": trk.id,    # or trk.track_id depending on your implementation
          "state": "confirmed" if is_confirmed else "tentative",
          "hits": trk.hits,
          "time_since_update": trk.time_since_update,
          "age": trk.age
      })

      all_data.append((detections.copy(), visual_tracks))

  run_interactive_viewer(all_data, frame_size)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Simulate OC-SORT tracking.')
  parser.add_argument('--dropout', type=int, default=0,
                      help='Number of consecutive frames to drop detections (simulate missed detection).')
  args = parser.parse_args()
  main(args.dropout)
