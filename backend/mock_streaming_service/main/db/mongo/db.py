from pymongo import MongoClient # type: ignore
from pymongo.server_api import ServerApi #type: ignore

from .db_collection import Collection

from ...exceptions.exceptions import QueryException, CollectionException

from ...utils.env import Env
 
# Username: brewern5_db_user
# pw: R9vnOR4x8SjFqW3U

class Db:
    def __init__(self):
        self.env = Env()
        self.uri = self.env.get_db_uri()
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))
        self.db = self.client['Tracks']

        self.collections = {}
        # Track db status 
        self.open = True

        try:
            self.client.admin.command('ping')
            print("Connection Established With Database.")
        except Exception as e:
            print("Could not connect to DB!")
            print(e)

        self._create_collections_map()

    """
        Collections
    """

    def _create_collections_map(self):
        collection_list = self.db.list_collections()

        for collection in collection_list: 
            #Create a collection map with the key as the name and the collection obj as its value
            mongo_collection = self.db[collection['name']]  # Get the actual MongoDB collection
            self.collections[collection['name']] = (Collection(mongo_collection)) 

    def _find_collection(self, collection_name):
        target_collection= self.collections.get(collection_name, None)
        if(target_collection == None):
            raise CollectionException("Collection not Found!", collection_name)
        else:
            return target_collection
    

    """
        Collection get based methods
    """


    def get_from_collection(self, collection_name, query_item):
        target_collection = self._find_collection(collection_name)

        # TODO: Create dynamic get methods 
        item = target_collection.get_by_name(query_item)
        if(item == None):
            raise QueryException("Item Not Found", query_item)
        
        return item
    
    def get_newest_from_collection(self, collection_name):
        target_collection = self._find_collection(collection_name)

        items = target_collection.get_most_recent(10)
        if(items == None):
            raise QueryException("Items Not Found")
        
        print('\n\n')
        print(type(items))
        
        return items

    """
        Collection post based methods
    """


    def post_to_collection(self, collection_name, post_item):

        target_collection = self._find_collection(collection_name)

        target_collection.post_item(post_item)
