from pipeline.index_documents import run_index_pipeline
import os
if __name__ == "__main__":
    print("📁 Working directory:", os.getcwd())
    run_index_pipeline()
