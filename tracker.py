from player import Player
from utils import calculate_iou
from config import FEATURE_SIMILARITY_THRESHOLD, IOU_TRACKING_THRESHOLD, MAX_LOST_FRAMES, SHORT_LIVED_FRAME_THRESHOLD

class Tracker:
    def __init__(self, extractor):
        self.players = {}
        self.extractor = extractor
        self.next_id = 0

    def update(self, frame, detections, frame_idx):
        detection_feats = [self.extractor.extract(frame, bbox) for bbox in detections]
        matched_det = set()
        used_ids = set()

        for i, (bbox, feat) in enumerate(zip(detections, detection_feats)):
            if feat is None:
                continue
            best_id, max_sim = -1, -1
            for pid, player in self.players.items():
                if pid in used_ids:
                    continue
                sim = self.extractor.compare(player.features, feat)
                if sim > max_sim:
                    best_id, max_sim = pid, sim
            if max_sim >= FEATURE_SIMILARITY_THRESHOLD:
                self.players[best_id].update(bbox, feat, frame_idx)
                matched_det.add(i)
                used_ids.add(best_id)

        for i, (bbox, feat) in enumerate(zip(detections, detection_feats)):
            if i in matched_det or feat is None:
                continue
            best_id, best_iou = -1, 0
            for pid, player in self.players.items():
                if pid in used_ids:
                    continue
                iou = calculate_iou(player.bbox, bbox)
                if iou > best_iou:
                    best_id, best_iou = pid, iou
            if best_iou >= IOU_TRACKING_THRESHOLD:
                sim = self.extractor.compare(self.players[best_id].features, feat)
                if sim > 0.3:
                    self.players[best_id].update(bbox, feat, frame_idx)
                    matched_det.add(i)
                    used_ids.add(best_id)

        for i, (bbox, feat) in enumerate(zip(detections, detection_feats)):
            if i in matched_det or feat is None:
                continue
            self.players[self.next_id] = Player(self.next_id, bbox, frame_idx, feat)
            used_ids.add(self.next_id)
            self.next_id += 1

        lost_ids = []
        for pid, player in self.players.items():
            if frame_idx - player.last_seen > MAX_LOST_FRAMES:
                if player.duration <= SHORT_LIVED_FRAME_THRESHOLD:
                    print(f"[INFO] Discarded short-lived ID {pid} (seen {player.duration} frames)")
                else:
                    print(f"[INFO] Lost long-term ID {pid} after {player.duration} frames")
                lost_ids.append(pid)
        for pid in lost_ids:
            del self.players[pid]

        return self.players
