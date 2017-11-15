from pymongo import MongoClient
"""
Helper Functions for MongoDB.

This is created such that I can handle errors in the event
that the game is running offline (mostly for testing purposes).

"""
class Mongo:
    def __init__(self):
        self.URI = None
        self.db = None
        self.connected = False

    def initiate(self,filepath):
        """
        Initiates a connection to the mongo instance.
        This is needed to store our results onto a database.
        Our webviewer will retrieve contents from there in order
        to display analytics related to slowpoke.
        """
        try:
            mongo = MongoClient(filepath)
            self.db = mongo.zephyr
            print(self.db)
            self.connected = True
            print("Successfully connected to Mongo.")
        except Exception as e:
            print(e)
            print("Warning: Slowpoke is not currently connected to a mongo instance.")

    def write(self, collection, entry):
        if self.connected:
            mongo_id = self.db[collection].insert(entry)
            return mongo_id
        else:
            return False
    
    def update(self, collection, mongo_id, entry):
        if self.connected:
            self.db[collection].update_one({'_id':mongo_id}, {"$set": entry}, upsert=False)

    def checkPlayerExists(self, player_id):
        if self.db['players'].find({'_id': player_id}).count() > 0:
            return True
        else:
            return False