#!/usr/bin/env bash

CSV="urls.csv"

# Skip header and iterate over each URL
tail -n +2 "$CSV" | while IFS= read -r url; do
  # Skip empty lines
  [[ -z "$url" ]] && continue

  echo "Downloading: $url"
  wget "$url"
done

