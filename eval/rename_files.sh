#!/bin/bash

# Script to rename files in results folder to match demo_color.m format
# label_*.png -> GT_*.png
# output_*.png -> Dem_*.png

RESULTS_DIR="results"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: $RESULTS_DIR directory not found!"
    exit 1
fi

# Process all subdirectories
for subdir in "$RESULTS_DIR"/*; do
    if [ -d "$subdir" ]; then
        echo "Processing directory: $subdir"
        
        # Process all Scene subdirectories
        for scene_dir in "$subdir"/*; do
            if [ -d "$scene_dir" ]; then
                echo "  Processing scene: $scene_dir"
                
                # Rename label_*.png to GT_*.png
                if [ -f "$scene_dir/label_0.png" ]; then
                    mv "$scene_dir/label_0.png" "$scene_dir/GT_0.png"
                    echo "    Renamed label_0.png -> GT_0.png"
                fi
                if [ -f "$scene_dir/label_45.png" ]; then
                    mv "$scene_dir/label_45.png" "$scene_dir/GT_45.png"
                    echo "    Renamed label_45.png -> GT_45.png"
                fi
                if [ -f "$scene_dir/label_90.png" ]; then
                    mv "$scene_dir/label_90.png" "$scene_dir/GT_90.png"
                    echo "    Renamed label_90.png -> GT_90.png"
                fi
                if [ -f "$scene_dir/label_135.png" ]; then
                    mv "$scene_dir/label_135.png" "$scene_dir/GT_135.png"
                    echo "    Renamed label_135.png -> GT_135.png"
                fi
                
                # Rename output_*.png to Dem_*.png
                if [ -f "$scene_dir/output_0.png" ]; then
                    mv "$scene_dir/output_0.png" "$scene_dir/Dem_0.png"
                    echo "    Renamed output_0.png -> Dem_0.png"
                fi
                if [ -f "$scene_dir/output_45.png" ]; then
                    mv "$scene_dir/output_45.png" "$scene_dir/Dem_45.png"
                    echo "    Renamed output_45.png -> Dem_45.png"
                fi
                if [ -f "$scene_dir/output_90.png" ]; then
                    mv "$scene_dir/output_90.png" "$scene_dir/Dem_90.png"
                    echo "    Renamed output_90.png -> Dem_90.png"
                fi
                if [ -f "$scene_dir/output_135.png" ]; then
                    mv "$scene_dir/output_135.png" "$scene_dir/Dem_135.png"
                    echo "    Renamed output_135.png -> Dem_135.png"
                fi
            fi
        done
    fi
done

echo "File renaming completed!"

