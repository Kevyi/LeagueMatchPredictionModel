import requests
import json
from datetime import datetime, timedelta, timezone
import endpoints as APIs

queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"] #Challenger/Grandmasteres/Masters requires still.

puuid = "_5RGhNXlzt3Fk6_K3ZE6UkEOBhXjvisoC7-HDq-3sDthWb2K8p3eYtKDrjiyQxfL7THIBGYbDYikhQ"
#-----------------------------------------------------------------------------------------------------------------
# Get current UTC date
today_utc = datetime.now(timezone.utc).date()
# Calculate yesterday's date
yesterday_date = today_utc - timedelta(days=4)
# Get midnight (start) of yesterday as a timezone-aware datetime
start_of_yesterday = datetime(yesterday_date.year, yesterday_date.month, yesterday_date.day, tzinfo=timezone.utc)

# Convert to epoch timestamp in seconds
start_time = int(start_of_yesterday.timestamp())
#-----------------------------------------------------------------------------------------------------------------


#Gets matches from yesterday to now. Ranked Flexed queue.                   
# APIs.getMatchesFromPlayer(region = "americas", puuid = puuid, startTime = start_time, count = 20, queue = 440, matchType = "ranked")
# APIs.getRankedUsers(queue = queue_types[0], tier = tiers[-1], division = divisions[0], page = 1, region = "na1")

APIs.getMatchStats(region = "americas", matchID = "NA1_5321872731")
#Use the champion id and not the name, use to determine pytorch winrate.