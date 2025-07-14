import cv2
from ultralytics import YOLO
from config import *
from feature_extractor import ReIDFeatureExtractor
from tracker import Tracker

def main():
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    model = YOLO(YOLO_MODEL_PATH)
    class_names = model.model.names
    player_cls = next((k for k, v in class_names.items() if v.lower() == PLAYER_CLASS_NAME), None)

    extractor = ReIDFeatureExtractor(REID_MODEL_PATH)
    tracker = Tracker(extractor)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        result = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)[0]
        detections = [
            box.cpu().numpy()
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls)
            if int(cls.item()) == player_cls
        ]

        active_players = tracker.update(frame, detections, frame_idx)

        for pid, player in active_players.items():
            x1, y1, x2, y2 = map(int, player.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        cv2.imshow("ReID Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Output saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
