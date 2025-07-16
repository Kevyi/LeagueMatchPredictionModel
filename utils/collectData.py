import json
import os
import endpoints as APIs
import sys
import time

queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"] #Challenger/Grandmasteres/Masters requires still.

def getAllChallengers():

    start_time = time.time() # Record the start time
    players = []
    page = 1
    while True:
        playerList, status_code = APIs.getRankedUsers(
            queue = "RANKED_SOLO_5x5", 
            tier = "CHALLENGER", 
            division = divisions[0], 
            page = page, 
            region = "na1"
        )

        #Breaks out of statement if we have an empty list, we exhausted all players in this rank.
        # print(sys.getsizeof(playerList))
        if playerList:
            for player in playerList:
                players.append(player["puuid"])
        else:
            break

        page += 1

    print(f"Byte Size: {sys.getsizeof(players)}")
    print(f"Player Amount: {len(players)}")
    print(f"Player Amount (HASHSET): {len(set(players))}")   
    print(f"Page Amount: {page}")
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def getAllGrandmasters():

    start_time = time.time() # Record the start time
    players = []
    page = 1
    while True:
        playerList, status_code = APIs.getRankedUsers(
            queue = "RANKED_SOLO_5x5", 
            tier = "GRANDMASTER", 
            division = divisions[0], 
            page = page, 
            region = "na1"
        )

        #Breaks out of statement if we have an empty list, we exhausted all players in this rank.
        # print(sys.getsizeof(playerList))
        if playerList:
            for player in playerList:
                players.append(player["puuid"])
        else:
            break

        page += 1

    print(f"Byte Size: {sys.getsizeof(players)}")
    print(f"Player Amount: {len(players)}")
    print(f"Player Amount (HASHSET): {len(set(players))}")   
    print(f"Page Amount: {page}")
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def getAllMasters():

    start_time = time.time() # Record the start time
    players = []
    page = 1
    while True:
        playerList, status_code = APIs.getRankedUsers(
            queue = "RANKED_SOLO_5x5", 
            tier = "MASTER", 
            division = divisions[0], 
            page = page, 
            region = "na1"
        )

        #Breaks out of statement if we have an empty list, we exhausted all players in this rank.
        # print(sys.getsizeof(playerList))
        if playerList:
            for player in playerList:
                players.append(player["puuid"])
        else:
            break

        page += 1

    print(f"Byte Size: {sys.getsizeof(players)}")
    print(f"Player Amount: {len(players)}")
    print(f"Player Amount (HASHSET): {len(set(players))}")   
    print(f"Page Amount: {page}")
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

def getAllDiamonds():

    start_time = time.time() # Record the start time
    players = []
    page = 1
    divisionIndex = 0
    while True:
        playerList, status_code = APIs.getRankedUsers(
            queue = "RANKED_SOLO_5x5", 
            tier = "DIAMOND", 
            division = divisions[divisionIndex], 
            page = page, 
            region = "na1"
        )

        if status_code == 429:
            time.sleep(10)
            continue

        #Breaks out of statement if we have an empty list and we exhausted all divisions, we exhausted all players in this rank.
        if playerList:
            for player in playerList:
                players.append(player["puuid"])
                print(player["puuid"])
        else:
            #Means division = "IV", the last division.
            if divisionIndex == 3: 
                break

            divisionIndex += 1

        page += 1

    print(f"Byte Size: {sys.getsizeof(players)}")
    print(f"Player Amount: {len(players)}")
    print(f"Player Amount (HASHSET): {len(set(players))}")   
    print(f"Page Amount: {page}")
    end_time = time.time() # Record the end time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

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