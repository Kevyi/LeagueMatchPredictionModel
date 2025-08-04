from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
import os

load_dotenv() 
db_uri = os.getenv("db_uri")

def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:

    def __init__(self):
        #Like a Singleton. Enacts DB.
        try: 
            self.client = MongoClient(db_uri)
            self.client.admin.command("ping")
            self.db_leagueData = self.client.get_database("leagueData")
            self.collectionNames = ["trainingMatchData", "models", "validationMatchData", "yesterdayMatchesData"]

            self.instantiateCollections()

            self.trainingCollection = self.db_leagueData.get_collection("trainingMatchData")
            self.modelsCollection = self.db_leagueData.get_collection("models")
            self.validationCollection = self.db_leagueData.get_collection("validationMatchData")
            self.yesterdayCollection = self.db_leagueData.get_collection("yesterdayMatchesData")

            #Deleted all yesterday's matches ID after one day (a little bit less). Generate uniqueness for their ids.
            # self.yesterdayCollection.create_index(
            #     [("createdAt", 1)],
            #     expireAfterSeconds=86400, #Just make it high for now.
            #     name="ttl_createdAt_1d"
            # )

            #ADD COLLECTION INDEX TO trainingCollection to filter by date, for new data.

            self.yesterdayCollection.create_index(
                "matchId", unique = True
            )

            print("Database connected successfully")
        except Exception as e:
            print(f"Failed to connect to Database. Exception: {e}")

    def instantiateCollections(self):
        for collectionName in self.collectionNames:
            if collectionName not in self.db_leagueData.list_collection_names():
                try:
                        # Option A: explicitly create an empty collection
                        self.db_leagueData.create_collection(collectionName)
                        print(f"Created empty collection '{collectionName}' in database '{self.db_leagueData}'.")
                except errors.CollectionInvalid:
                    # race-condition safety: someone else made it
                    print(f"Collection '{collectionName}' already exists.")
            else:
                print(f"Collection '{collectionName}' already exists in '{self.db_leagueData}'.")

    def insertData(self, data, dataType = "training"):
        match dataType:
            case "training":
                self.trainingCollection.insert_one(data)
            case "validation":
                self.validationCollection.insert_one(data)
            case "yesterdayMatches":
                try:
                    self.yesterdayCollection.insert_many(data, ordered=False)
                except BulkWriteError as duplicateError:
                    pass
                except Exception as e:
                    print(f"database error: {e}")
    
    def insertYesterdayMatch(self, data):
        try:
            self.yesterdayCollection.insert_many(data, ordered=False)
        except BulkWriteError as duplicateError:
            pass
        except Exception as e:
            print(f"database error: {e}")

    #Change getData later to determine the batch size per dataset.
    def getData(self, dataType = "training"):
        #Depending on dataType
        collection = None
        batch_size = 4096
        
        match dataType:
            case "training":
                collection = self.trainingCollection
            case "validation":
                collection = self.validationCollection
            case "models":
                collection = self.modelsCollection
            case "yesterdayMatches":
                collection = self.yesterdayCollection

        #Finds all documents.
        cursor = collection.find({}) #.batch_size(BATCH_SIZE)

        res = list(cursor)

        return res

db = Database()




























