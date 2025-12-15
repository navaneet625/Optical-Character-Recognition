import shutil
import os
import glob

def clean_project():
    print("Starting Project Cleanup...")
    
    # 1. Define paths to clean
    paths_to_remove = [
        "data/images",
        "data/labels",
        "data/labels.txt",
        "__pycache__",
        "models/__pycache__",
        "configs/__pycache__",
        "data/__pycache__",
        "error_analysis.png",
        "ocr.onnx"
    ]
    
    # 2. Remove Directories and Files
    for path in paths_to_remove:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"Removed directory: {path}")
                else:
                    os.remove(path)
                    print(f"Removed file: {path}")
            except Exception as e:
                print(f"Failed to remove {path}: {e}")
        else:
            print(f"Skipped (not found): {path}")

    # 3. Recursive cleanup for .pyc and .DS_Store
    print("\nDeep cleaning .pyc and .DS_Store files...")
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".pyc") or file == ".DS_Store":
                full_path = os.path.join(root, file)
                try:
                    os.remove(full_path)
                except Exception as e:
                    pass
    
    print("\nCleanup Complete!")

if __name__ == "__main__":
    confirmation = input("This will delete all generated images and labels. Are you sure? (y/n): ")
    if confirmation.lower() == 'y':
        clean_project()
    else:
        print("Operation cancelled.")
