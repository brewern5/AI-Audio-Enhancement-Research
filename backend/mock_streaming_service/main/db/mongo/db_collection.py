class Collection:

    def __init__(self, collection):
        self.collection = collection

    def find_one(self, *args, **kwargs):
        return self.collection.find_one(*args, **kwargs)

    def get_all(self, db):
        pass

    def get_by_id(self, track_id):
        return self.find_one(track_id)

    def get_by_name(self, name):
        query = {'name': name}
        result = self.find_one(query)
        return result