#!/usr/bin/env bash

shopt -s nullglob

for file in *; do
  # Skip directories
  [[ -d "$file" ]] && continue

  # Only process files that contain "FOMC"
  if [[ "$file" == *FOMC* ]]; then
    newname="${file#*FOMC}"
    newname="FOMC${newname}"

    # Only rename if the name actually changes
    if [[ "$file" != "$newname" ]]; then
      mv -i -- "$file" "$newname"
    fi
  fi
done

