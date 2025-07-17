import json
import os
from datetime import datetime, timedelta, timezone
import endpoints as APIs
import sys
import time

queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"] #Challenger/Grandmasteres/Masters require still.

puuidFile = "puuid.json"
yesterdayMatches = "yesterdayMatches.json"

# If puuid file doesn't exist, create it with empty data. Runs at beginning from import.
if not os.path.exists(puuidFile):
    with open(puuidFile, 'w') as f:
        json.dump([], f)  # or [] if you want an empty list

# # Reset or creates yesterdayMatches json. --> Takes too many api calls.
# with open(yesterdayMatches, 'w') as f:
#     json.dump([], f)  # or [] if you want an empty list

def getRankedPlayers(tier : str):

    start_time = time.time() # Record the start time
    players = set()
    page = 1
    divisionIndex = 0

    #Ensures proper parameters.
    if tier not in tiers: return

    while True:
        playerList, status_code = APIs.getRankedUsers(
            queue = "RANKED_SOLO_5x5", 
            tier = tier, 
            division = divisions[divisionIndex], 
            page = page, 
            region = "na1"
        )

        #Breaks out of statement if we have an empty list and we exhausted all divisions, we exhausted all players in this rank.
        if playerList:
            for player in playerList:
                players.add(player["puuid"])
        else:
            #Means division = "IV", the last division has been fulfilled OR top ranked have been fulfilled.
            if divisionIndex == 3 or tier in ["MASTER", "GRANDMASTER", "CHALLENGER"]: 
                break

            divisionIndex += 1

        page += 1
    
    for player in players:
        getYesterdayPlayerMatches(player)
        #Buffer calls.
        time.sleep(0.2)

#--------------------
    #Store in JSON temporarily. Move to SQL later.
    with open(puuidFile, 'r') as f:
        currentPuuids = json.load(f)
    
    currentPuuids.extend(players)
    currentPuuids = list(set(currentPuuids)) #Ensure unique items.
    
    with open(puuidFile, 'w') as f:
        json.dump(currentPuuids, f, indent=4)
#--------------------

    print(f"Byte Size: {sys.getsizeof(players)}")
    print(f"Player Amount: {len(players)}")
    print(f"Page Amount: {page}")
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")



def getYesterdayPlayerMatches(puuid : str):

    # Get current UTC date
    today_utc = datetime.now(timezone.utc).date()

    # Calculate yesterday's date
    yesterday_date = today_utc - timedelta(days=1)

    # Get midnight (start) of yesterday as a timezone-aware datetime
    start_of_yesterday = datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, tzinfo=timezone.utc)

    # Convert to epoch timestamp in seconds
    start_time = int(start_of_yesterday.timestamp())

    matches, status_code = APIs.getMatchesFromPlayer(region = "americas", puuid = puuid, startTime = start_time, count = 20, queue = 440, matchType = "ranked")

    #--------------------
    #Store in JSON temporarily. Move to SQL later.
    with open(yesterdayMatches, 'r') as f:
        currentMatches = json.load(f)
    
    currentMatches.extend(matches)
    currentMatches = list(set(currentMatches)) #Ensure unique items.
    
    with open(yesterdayMatches, 'w') as f:
        json.dump(currentMatches, f, indent = 4)

    print(json.dumps(matches, indent = 4))  
    pass
























def responseHandler(response : int):
    """
    HTTP Status Code	Reason
    400	Bad request
    401	Unauthorized
    403	Forbidden
    404	Data not found
    405	Method not allowed
    415	Unsupported media type
    429	Rate limit exceeded
    500	Internal server error
    502	Bad gateway
    503	Service unavailable
    504	Gateway timeout
    """

    pass