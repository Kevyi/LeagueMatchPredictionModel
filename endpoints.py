from dotenv import load_dotenv
from utils.collectData import writeChampion
import time
import requests
import json
import os

load_dotenv() 
api_key = os.getenv("RIOT_KEY")
api_key2 = os.getenv("RIOT_KEY2")
api_key3 = os.getenv("RIOT_KEY3")
keys = [api_key, api_key2, api_key3]
session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key
}

servers = ["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2", "me1","na1","oc1","ru", "sg2", "tr1", "tw2", "vn2"]
regions = ["americas", "asia", "europe", "sea"]
queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"]

#Store inside SQL database, have a recheck method weekly to check any changes in attributes.
def getRankedUsers(queue : str, tier : str, division : str, page : int, server: str = "na1"):

    url = f"https://{server}.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page}"

    response = attemptRequest(url)

    return response.json() , response.status_code

#remember to check for match duplicates. 420 = soloq, 440 = ranked flex
def getMatchesFromPlayer(region : str, puuid : str, startTime : int, count : int,  queue : int = 420, matchType : str = "ranked"):
    #everything url: https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime=1000&endTime=0&queue=0&type=ranked&start=4&count=20
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={startTime}&queue={queue}&type={matchType}&count={count}"

    response = attemptRequest(url)

    return response.json(), response.status_code

def getMatchStats(matchId : str, region : str = "americas",):
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{matchId}?api_key={api_key}"

    response = attemptRequest(url)
    
    return response.json(), response.status_code


def attemptRequest(url : str):
    max_retries = 10
    delay = 15  # seconds between retries
    keysIndex = 0

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
    #Attempts to get a connection request using 3 API keys. Could change code sequence to check 3 api keys at once. IP restricted.
        #use iterator.
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, headers = headers, timeout=5)
            response.raise_for_status()  # raise for 4xx or 5xx errors
            return response
        except requests.exceptions.ReadTimeout:
            print(f"[{attempt}] Timeout. Retrying in {delay}s...")
        except requests.exceptions.RequestException as e:

            if e.response.status_code == 429:
                print("Attempting different API key.")
                keysIndex += 1
                session.headers.update({"X-Riot-Token": keys[keysIndex % len(keys)]})
                print(session.headers)

            print(f"[{attempt}] Request failed: {e}. Retrying in {delay}s...")
        time.sleep(delay)
    else:
        print("❌ Failed after retries.")

def getMatchTimeLine():
    pass
