import json
import os
from datetime import datetime, timedelta, timezone
import endpoints as APIs
import sys
import time

queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"] #Challenger/Grandmasters/Masters require still.

puuidFile = "puuid.json"
yesterdayMatchesFile = "yesterdayMatches.json"
trainingDataFile = "trainingData.json"

# Checks if files below are already made, if not make it.
if not os.path.exists(puuidFile):
    with open(puuidFile, 'w') as f:
        json.dump({}, f)  # or [] if you want an empty list

if not os.path.exists(trainingDataFile):
    with open(trainingDataFile, 'w') as f:
        json.dump([], f)  # or [] if you want an empty list

# # Reset or creates yesterdayMatchesFile json. --> Takes too many api calls.
if not os.path.exists(yesterdayMatchesFile):
    with open(yesterdayMatchesFile, 'w') as f:
        json.dump([], f)  # or [] if you want an empty list

def getAllYesterdayPlayerMatches():
    puuidFile = "puuid.json"

    #Did challenger and prob some grandmaster lastnight. Switch to master or diamond for more matches.
    with open(puuidFile, "r") as f:
        puuids = json.load(f)
        for id in puuids["CHALLENGER"]:
            getYesterdayPlayerMatches(id)
        for id in puuids["GRANDMASTER"]:
            getYesterdayPlayerMatches(id)
        for id in puuids["MASTER"]:
            getYesterdayPlayerMatches(id)
        for id in puuids["DIAMOND"]:
            getYesterdayPlayerMatches(id)

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
            server = "na1"
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
    
    count = 0

    #Store in JSON temporarily. Move to SQL later.
    with open(puuidFile, 'r') as f:
        currentPuuids = json.load(f)
    
    currentPuuids.setdefault(tier, []).extend(players)
    currentPuuids[tier] = list(set(currentPuuids[tier])) #Ensure unique items. Unneeded?
    
    with open(puuidFile, 'w') as f:
        json.dump(currentPuuids, f, indent=4)

    print(f"Byte Size: {sys.getsizeof(players)}")
    print(f"Player Amount: {len(players)}")
    print(f"Page Amount: {page}")
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def getYesterdayPlayerMatches(puuid : str):

    # Get current UTC date
    today_utc = datetime.now(timezone.utc).date()

    # Calculate yesterday's date -->
    yesterday_date = today_utc - timedelta(days=1) 

    # Get midnight (start) of yesterday as a timezone-aware datetime
    start_of_yesterday = datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, tzinfo=timezone.utc)

    # Convert to epoch timestamp in seconds
    start_time = int(start_of_yesterday.timestamp())

    matches, status_code = APIs.getMatchesFromPlayer(region = "americas", 
        puuid = puuid, 
        startTime = start_time, 
        count = 100, 
        queue = 440, 
        matchType = "ranked"
    )

    #Store in JSON temporarily. Move to S3 buckets later.
    with open(yesterdayMatchesFile, 'r') as f:
        currentMatches = json.load(f)
    
    currentMatches.extend(matches)
    currentMatches = list(set(currentMatches)) #Ensure unique items.
    
    with open(yesterdayMatchesFile, 'w') as f:
        json.dump(currentMatches, f, indent = 4)

def getMatchDetails(matchId : str):

    matchInformation, status_code = APIs.getMatchStats(matchId = matchId, region = "americas") 
    gameStatus = matchInformation['info']['endOfGameResult']

    #Checks if game has actually finished (played through). 
    if gameStatus != "GameComplete": return

    players = matchInformation['info']['participants']
    additionalTeamInfo = matchInformation['info']["teams"]
    teams = {100 : {}, 200: {}, "matchId": matchId}


    #records champions into JSON file.
    for player in players:
        #Records Champions.
        writeChampion(player["championId"], player["championName"])
        #Adds to team dictionary.
        teams[player["teamId"]][player["individualPosition"]] = player["championId"]

    #Finds which team won.
    if additionalTeamInfo[0]["win"]:
        teams["win"] = additionalTeamInfo[0]["teamId"]
    else:
        teams["win"] = additionalTeamInfo[1]["teamId"]

    #Store in JSON temporarily.
    with open(trainingDataFile, 'r') as f:
        currentTrainingData = json.load(f)
    
    currentTrainingData.append(teams)
    
    with open(trainingDataFile, 'w') as f:
        json.dump(currentTrainingData, f, indent = 4)

def writeChampion(championId : int, championName : str):

    fileName = "champions.json"

    # If file doesn't exist, create it with empty data
    if not os.path.exists(fileName):
        with open(fileName, 'w') as f:
            json.dump({}, f)  # or [] if you want an empty list

    # 1. Read the JSON file
    with open(fileName, 'r') as f:
        data = json.load(f)

    # # 2. Modify the data if champion not found. --> Sometimes adds multiple same key-value pairs.
    if championId not in data:
        data[championId] = championName

    # 3. Write it back to the file
    with open(fileName, 'w') as f:
        json.dump(data, f, indent=4)  # 'indent' makes it pretty-printed


























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