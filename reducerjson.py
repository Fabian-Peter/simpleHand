import json

# Specify the output JSON file path
output_json_path = "/home/fabian/simpleHand/dataset/eval_1.json"

# Base directory for the paths
base_path = "/home/fabian/simpleHand/data/FreiHAND/evaluation/rgb"

# Range of file numbers
start_number = 0
end_number = 3959

# Generate the list of file paths
file_paths = [
    f"{base_path}/{str(i).zfill(8)}.jpg" for i in range(start_number, end_number + 1)
]

# Save the list to a JSON file
with open(output_json_path, "w") as f:
    json.dump(file_paths, f, indent=4)

print(f"JSON file created with {len(file_paths)} file paths.")
