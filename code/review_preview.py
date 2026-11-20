"""Render a close-up for the selected observation on its exact source frame."""
import cv2


def focused_tooth_preview(source, observation, color=(70, 230, 125)):
    if source is None or not source.is_file():
        return None
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, observation['source_frame']):
            return None
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        return None

    height, width = frame.shape[:2]
    left, top, right, bottom = observation['bbox_xyxy_normalized']
    x1, y1, x2, y2 = round(left * width), round(top * height), round(right * width), round(bottom * height)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    pad_x, pad_y = max((x2 - x1) * 2, width // 10), max(y2 - y1, height // 12)
    crop_left, crop_top = max(0, x1 - pad_x), max(0, y1 - pad_y)
    crop = frame[crop_top:min(height, y2 + pad_y), crop_left:min(width, x2 + pad_x)].copy()
    thickness = max(2, crop.shape[1] // 220)
    a, b = (x1 - crop_left, y1 - crop_top), (x2 - crop_left - 1, y2 - crop_top - 1)
    cv2.rectangle(crop, a, b, (15, 15, 15), thickness + 3)
    cv2.rectangle(crop, a, b, color, thickness)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
