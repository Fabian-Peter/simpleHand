import os

# Specify the folder containing the files
folder_path = "/home/fabian/simpleHand/data/FreiHAND/training/rgb"

# Starting point (inclusive)
start_number = 5000

# Loop through the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a jpg or json and its number is >= start_number
    if filename.endswith((".jpg", ".json")):
        # Extract the numeric part from the filename
        file_number = int(filename.split(".")[0])
        if file_number >= start_number:
            # Build the full path and remove the file
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Removed: {file_path}")
