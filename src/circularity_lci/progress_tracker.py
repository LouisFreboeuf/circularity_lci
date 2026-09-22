import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track progress across runs"""

    def __init__(self, project_name):
        self.project_name = project_name
        self.json_dir = os.path.join("tracker", "json")
        os.makedirs(self.json_dir, exist_ok=True)  # Create the json directory if it doesn't exist
        self.progress_file = os.path.join(
            self.json_dir,
            f"{project_name}_progress.json"
        )
        self.progress_data = self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error loading progress file: %s", e)
                return {}
        return {}

    def save_progress(self):
        with open(self.progress_file, "w") as f:
            json.dump(self.progress_data, f, indent=2)

    def update_step(self, step, status, data=None, execution_time=None):
        self.progress_data[step] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time,
            "data_summary": data
        }
        self.save_progress()

    def is_step_completed(self, step):
        return step in self.progress_data and self.progress_data[step]["status"] == "completed"

    def print_summary(self):
        print("\n=== Progress Summary ===")
        for step, data in self.progress_data.items():
            print(f"{step}: {data.get('status')} at {data.get('timestamp')} (time: {data.get('execution_time')})")

    def log_summary(self):
        logger.info("Progress summary:")
        for step, data in self.progress_data.items():
            logger.info(
                "%s: %s at %s (time: %s)",
                step,
                data.get('status'),
                data.get('timestamp'),
                data.get('execution_time'),
            )