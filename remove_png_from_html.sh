#!/bin/bash

# Usage: ./remove_pngs_from_html.sh yourfile.html

if [ $# -ne 1 ]; then
  echo "Usage: $0 <html_file>"
  exit 1
fi

HTML_FILE="$1"

# Find all .png file paths in the HTML file (case-insensitive), strip quotes and possible relative paths
PNG_FILES=$(grep -oiE 'src=["'\'']?[^"'\'']+\.png' "$HTML_FILE" | sed -E 's/src=["'\'']?//' | sort -u)

if [ -z "$PNG_FILES" ]; then
  echo "No PNG files found in $HTML_FILE."
  exit 0
fi

# Loop through the found PNG files and remove them using git rm
for file in $PNG_FILES; do
  if [ -f "$file" ]; then
    echo "Removing $file"
    git rm "$file"
  else
    echo "File not found or not tracked by git: $file"
  fi
done
git rm $1

