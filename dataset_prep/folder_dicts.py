import os
import json
from collections import defaultdict, OrderedDict


# # Training data

# Create a defaultdict with set as the default value type
d=defaultdict(set)

# Walk through all directories and files in the specified path
for path,dirs,files in os.walk('./labeled-videos-Processed/Resized/Frames/'):
   # Extract the base name of the current directory and store the count of files in it
   d[os.path.basename(path)]=len(files)

# Remove the entry with an empty key, which corresponds to the base directory itself
d.pop("")

# Save the directory name and file count information into a JSON file called 'train.json'
with open('train.json', 'w') as fp:
    # Sort the items of the dictionary and write it to the JSON file with minimal indentation
    json.dump(OrderedDict(sorted(d.items())), fp, indent=0)