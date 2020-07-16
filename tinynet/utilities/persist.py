import json
class SummaryWriter():
    def __init__(self, filepath):
        self.filepath = filepath
        self.records = []

    def append(self, record):
        self.records.append(record)

    def write(self):
        with open(self.filepath) as f:
            f.write(json.dumps(self.records))