"""MongoDB helper functions for storing game results."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union


class Mongo:
    """MongoDB connection manager for storing game results."""

    def __init__(self) -> None:
        self.URI: Optional[str] = None
        self.db: Any = None
        self.connected: bool = False

    def initiate(self, filepath: str) -> None:
        """Connect to a MongoDB instance.

        Args:
            filepath: MongoDB URI connection string.
        """
        try:
            from pymongo import MongoClient

            mongo = MongoClient(filepath)
            self.db = mongo.zephyr
            self.connected = True
            print("Successfully connected to Mongo.")
        except Exception as e:
            self.connected = False
            print(e)
            print("Warning: Slowpoke is not currently connected to a mongo instance.")

    def write(self, collection: str, entry: Dict[str, Any]) -> Union[str, bool]:
        """Write a document to a MongoDB collection.

        Args:
            collection: Collection name.
            entry: Document to insert.

        Returns:
            Inserted document ID, or False if not connected.
        """
        if self.connected:
            mongo_id = self.db[collection].insert(entry)
            return mongo_id
        return False

    def update(self, collection: str, mongo_id: str, entry: Dict[str, Any]) -> None:
        """Update a document in a MongoDB collection.

        Args:
            collection: Collection name.
            mongo_id: Document ID to update.
            entry: Fields to update.
        """
        if self.connected:
            self.db[collection].update_one(
                {"_id": mongo_id}, {"$set": entry}, upsert=False
            )

    def check_player_exists(self, player_id: str) -> bool:
        """Check if a player exists in the database.

        Args:
            player_id: Player ID to check.

        Returns:
            True if the player exists.
        """
        if self.connected and self.db is not None:
            return self.db["players"].find({"_id": player_id}).count() > 0
        return False
