import subprocess
import time

def run_notebook(nb_path):
    args = [
        "jupyter",
        "nbconvert",
        "--to=notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=10000000",
        nb_path,
    ]

    print(f">> Running notebook {nb_path}")
    ts = time.time()
    subprocess.check_output(args)
    print(f">> Finished running notebook {nb_path}, elapsed={time.time() - ts:.3f}s")

if __name__ == "__main__":
    # Replace this with the path to your notebook
    notebook_path = "./YOLO_Fine_Tuning.ipynb"
    run_notebook(notebook_path)
