import requests
import os
import json
from datetime import datetime, timedelta, timezone
import endpoints as APIs
import utils.collectData as data

queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"] #Challenger/Grandmasteres/Masters requires still.

puuid = "_5RGhNXlzt3Fk6_K3ZE6UkEOBhXjvisoC7-HDq-3sDthWb2K8p3eYtKDrjiyQxfL7THIBGYbDYikhQ"

#Gets matches from yesterday to now. Ranked Flexed queue.                   
# APIs.getMatchesFromPlayer(region = "americas", puuid = puuid, startTime = start_time, count = 20, queue = 440, matchType = "ranked")
# APIs.getRankedUsers(queue = queue_types[0], tier = tiers[-1], division = divisions[0], page = 1, region = "na1")

# data.getRankedPlayers("CHALLENGER")
# data.getRankedPlayers("GRANDMASTER")
# data.getRankedPlayers("MASTER")
# data.getRankedPlayers("DIAMOND")
# data.getYesterdayPlayerMatches("cVvO4uv1pn_i88-lBhCBEIf1gcg5E21HkEOgC2nM54KAfthq1cYjUxBgoPqkY9zjkAR9EMW_pDkpdw")



data.getAllYesterdayPlayerMatches()


# matchesFile = "yesterdayMatchesTraining.json"
# count = 0

# with open(matchesFile, "r") as f:
#     matches = json.load(f)
#     for match in matches:
#         data.getMatchDetails(match)
#         count += 1
#         print(count)


# APIs.getMatchStats(region = "americas", matchID = "NA1_5325004163")
#Use the champion id and not the name, use to determine pytorch winrate.