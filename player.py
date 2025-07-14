class Player:
    def __init__(self, pid, bbox, frame_num, feat):
        self.id = pid
        self.bbox = bbox
        self.features = feat
        self.last_seen = frame_num
        self.start_frame = frame_num

    def update(self, bbox, feat, frame_num):
        self.bbox = bbox
        self.features = 0.8 * self.features + 0.2 * feat
        self.last_seen = frame_num

    @property
    def duration(self):
        return self.last_seen - self.start_frame + 1
