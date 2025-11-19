import pymongo #type: ignore

class Collection:

    def __init__(self, collection):
        self.collection = collection

    def _find_one(self, *args, **kwargs):
        return self.collection.find_one(*args, **kwargs)

    def get_most_recent(self, num):
        results = self.collection.find().sort("timestamp", pymongo.DESCENDING).limit(num)
        return list(results)

    """
        Get Methods
    """

    def get_by_id(self, track_id):
        return self._find_one(track_id)

    def get_by_name(self, name):
        query = {'name': name}
        result = self._find_one(query)
        return result
    
    """
        Post Methods
    """

    def post_item(self, item_data):
        self.collection.insert_one(item_data)
