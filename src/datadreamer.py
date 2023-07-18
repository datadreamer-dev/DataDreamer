import os
import threading


class DataDreamer:
    ctx = threading.local()

    def __init__(self, output_folder_path: str):
        if os.path.isfile(output_folder_path):
            raise ValueError(
                f"Expected path to folder, but file exists at path {output_folder_path}."
            )
        self.output_folder_path = output_folder_path

    def __enter__(self):
        if hasattr(DataDreamer.ctx, "steps"):
            raise RuntimeError("Cannot nest DataDreamer() context managers.")
        os.makedirs(self.output_folder_path)
        DataDreamer.ctx.output_folder_path = self.output_folder_path
        DataDreamer.ctx.steps = []

    def __exit__(self, exc_type, exc_value, exc_tb):
        DataDreamer.ctx = threading.local()
