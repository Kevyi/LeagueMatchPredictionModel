import json
import os

def writeChampion(championId : int, championName : str):

    fileName = "champions.json"

    # If file doesn't exist, create it with empty data
    if not os.path.exists(fileName):
        with open(fileName, 'w') as f:
            json.dump({}, f)  # or [] if you want an empty list

    # 1. Read the JSON file
    with open(fileName, 'r') as f:
        data = json.load(f)

    # 2. Modify the data if champion not found.
    if championId not in data:
        data[championId] = championName

    # 3. Write it back to the file
    with open(fileName, 'w') as f:
        json.dump(data, f, indent=4)  # 'indent' makes it pretty-printed



