from dotenv import load_dotenv
from utils.writeChampions import writeChampion
import requests
import json
import os

load_dotenv() 
api_key = os.getenv("RIOT_KEY")

servers = ["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2", "me1","na1","oc1","ru", "sg2", "tr1", "tw2", "vn2"]
regions = ["americas", "asia", "europe", "sea"]
queue_types = ["RANKED_SOLO_5x5","RANKED_FLEX_SR"]
tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"] #ranks
divisions = ["I", "II", "III", "IV"]

#Store inside SQL database, have a recheck method weekly to check any changes in attributes.
def getRankedUsers(queue : str, tier : str, division : str, page : int, region: str = "na1"):
    #explicitly cast page int to string.
    url = f"https://{region}.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page}&api_key={api_key}"

    response = requests.get(url)
    print(response.status_code)
    # print(json.dumps(response.json(), indent = 4))  
    return response.json() , response.status_code

#remember to check for match duplicates. 420 = soloq, 440 = ranked flex
def getMatchesFromPlayer(region : str, puuid : str, startTime : int, count : int,  queue : int = 420, matchType : str = "ranked"):
    #everything url: https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime=1000&endTime=0&queue=0&type=ranked&start=4&count=20&api_key={api_key}
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?startTime={startTime}&queue={queue}&type={matchType}&count={count}&api_key={api_key}"

    response = requests.get(url)

    print(json.dumps(response.json(), indent = 4))  

def getMatchStats(region : str, matchID : str):
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{matchID}?api_key={api_key}"

    response = requests.get(url)
    # print(json.dumps(response.json()['info']['participants'][0]["championName"], indent = 4))
    players = response.json()['info']['participants']

    for i in range(10):
        player = players[i]
        writeChampion(player["championId"], player["championName"])

def getMatchTimeLine():
    pass
