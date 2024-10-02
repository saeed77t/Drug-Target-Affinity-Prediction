#!/bin/bash

# Paths
aligned_protein_path="data/kiba/aln"
contact_map_dir="KibaContactMaps"
high_ram_list="high_ram_proteins.txt"

# Ensure the contact maps directory exists
mkdir -p "$contact_map_dir"

# Initialize or touch the high RAM list file
touch "$high_ram_list"

# Function to get RAM usage percentage
get_ram_usage() {
    mem_info=$(free -m)
    total_mem=$(echo "$mem_info" | awk '/^Mem:/ {print $2}')
    used_mem=$(echo "$mem_info" | awk '/^Mem:/ {print $3}')
    ram_usage_percent=$(awk "BEGIN {printf \"%.0f\", ($used_mem/$total_mem)*100}")
    echo $ram_usage_percent
}

# Function to process a single protein
process_protein() {
    local input_file="$1"
    local output_file="$2"
    local protein_name="$3"

    while true; do
        echo "Processing $input_file"

        # Run the Python script in the background
        python predict_contact_map.py "$input_file" "$output_file" &
        python_pid=$!

        # Monitor RAM usage every 5 seconds
        ram_exceeded=0
        while kill -0 "$python_pid" 2> /dev/null; do
            sleep 5  # Wait before checking RAM usage
            ram_usage=$(get_ram_usage)

            if [ "$ram_usage" -ge 95 ]; then
                echo "RAM usage is over 95% while processing $protein_name. Killing Python script."
                # Kill the Python process
                kill -9 "$python_pid"
                wait "$python_pid" 2>/dev/null
                # Ensure all related Python processes are terminated
                pkill -f "predict_contact_map.py $input_file $output_file"
                echo "$protein_name" >> "$high_ram_list"  # Add protein to high RAM list
                ram_exceeded=1
                break  # Exit the RAM monitoring loop to restart or skip the process
            fi
        done

        # Check if the Python script has exited successfully
        if ! kill -0 "$python_pid" 2> /dev/null; then
            if [ "$ram_exceeded" -eq 0 ]; then
                echo "Python script completed successfully for $protein_name."
                # Remove protein from high RAM list if it's there
                sed -i "/^$protein_name$/d" "$high_ram_list"
                break  # Exit the retry loop and proceed to the next file
            else
                echo "Processing of $protein_name exceeded RAM limit."
                break  # Exit the retry loop; will attempt later if in high RAM list
            fi
        fi

        # Optional: Add a short delay before retrying
        sleep 1
    done

    # Optional: Add a short delay to help with resource management
    sleep 1
}

# Read the high RAM list into an array
mapfile -t high_ram_proteins < "$high_ram_list"

# Process proteins not in the high RAM list
echo "Processing proteins not in high RAM list..."
file_list=$(ls "$aligned_protein_path"/*.aln)

for input_file in $file_list; do
    file_name=$(basename "$input_file")
    protein_name="${file_name%.aln}"
    output_file="$contact_map_dir/${protein_name}.npy"  # Remove .aln extension and add .npy

    if [ -f "$output_file" ]; then
        # Skip if the contact map already exists
        continue
    fi

    # Skip if protein is in the high RAM list
    if [[ " ${high_ram_proteins[@]} " =~ " ${protein_name} " ]]; then
        continue
    fi

    process_protein "$input_file" "$output_file" "$protein_name"
done

# Reattempt processing proteins in the high RAM list
echo "Reattempting proteins in high RAM list..."
for protein_name in "${high_ram_proteins[@]}"; do
    input_file="$aligned_protein_path/${protein_name}.aln"
    output_file="$contact_map_dir/${protein_name}.npy"

    if [ -f "$output_file" ]; then
        # Skip if the contact map already exists
        continue
    fi

    process_protein "$input_file" "$output_file" "$protein_name"
done

echo "All processing completed."

