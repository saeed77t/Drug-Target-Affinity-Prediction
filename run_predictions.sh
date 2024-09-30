#!/bin/bash

# Paths
aligned_protein_path="data/kiba/aln"
contact_map_dir="KibaContactMaps"

# Ensure the contact maps directory exists
mkdir -p "$contact_map_dir"

# Function to get RAM usage percentage
get_ram_usage() {
    mem_info=$(free -m)
    total_mem=$(echo "$mem_info" | awk '/^Mem:/ {print $2}')
    used_mem=$(echo "$mem_info" | awk '/^Mem:/ {print $3}')
    ram_usage_percent=$(awk "BEGIN {printf \"%.0f\", ($used_mem/$total_mem)*100}")
    echo $ram_usage_percent
}

# Iterate through each alignment file
file_list=$(ls "$aligned_protein_path"/*.aln)

for input_file in $file_list; do
    file_name=$(basename "$input_file")
    output_file="$contact_map_dir/${file_name%.aln}.npy"  # Remove .aln extension and add .npy

    if [ -f "$output_file" ]; then
        echo "$output_file already exists, skipping."
        continue  # Skip if the contact map already exists
    fi

    echo "Processing $input_file"

    # Loop to retry running the Python script if RAM usage exceeds threshold
    while true; do
        # Run the Python script in the background
        python predict_contact_map.py "$input_file" "$output_file" &
        python_pid=$!

        # Monitor RAM usage every 5 seconds
        while kill -0 "$python_pid" 2> /dev/null; do
            ram_usage=$(get_ram_usage)
            #echo "Current RAM usage: $ram_usage%"

            if [ "$ram_usage" -ge 92 ]; then
                echo "RAM usage is over 92%. Killing Python script."
                kill -9 "$python_pid"
                wait "$python_pid" 2>/dev/null
                echo "Restarting Python script."
                break
            fi

            sleep 5
        done

        # Check if the Python script has exited successfully
        if ! kill -0 "$python_pid" 2> /dev/null; then
            echo "Python script completed successfully."
            break  # Exit the retry loop and proceed to the next file
        fi

        # Optional: Add a short delay before retrying
        sleep 1
    done

    # Optional: Add a short delay to help with resource management
    sleep 1
done

