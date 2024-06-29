#!/bin/bash

# set folder
parent_folder="sato_tables"

for item in "$parent_folder"/*; do
	if [ -d "$item" ]; then
		for subdir in "$item"/*; do
        		# get folder name
        		subdir_name=$(basename "$subdir")	

			# init count
			counter=1
			# process all files
			for file in "$item"/"$subdir_name"/*; do
				# rename
				new_name="${item}/${subdir_name}/${subdir_name}_${counter}"
				mv "$file" "$new_name"
				# update count
				counter=$((counter + 1))
			done
		done
	fi
done

echo "All files have been renamed."

