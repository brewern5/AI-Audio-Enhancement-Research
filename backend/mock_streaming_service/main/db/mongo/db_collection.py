class Collection:

    def __init__(self, collection):
        self.collection = collection

    def _find_one(self, *args, **kwargs):
        return self.collection.find_one(*args, **kwargs)

    def get_all(self, db):
        pass

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
         
        posted = self.collection.insert_one(item_data)

        print(posted)

        return posted