import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Loaded " + __name__)
from pymongo import MongoClient

import os
class mongo_connect():

    def __init__(self,mongo_auth,collection):
        self.conn= mongo_auth
        self.collection=collection
        self.connect()
        logger.info("mongo initialized")

    def connect(self):
        maxTries=20
        trial=0
        while True:
            trial +=1
            try:
                logger.info("trying to connect to mongo")
                self.client = MongoClient(self.conn)
                self.db = self.client.get_database(self.collection)
                return True
            except Exception as e:
                print("unable to connect",str(e))
            
            if trial>maxTries:
                logger.error("mongo conn failed")
                return False
    
    def get_context_vars(self,user_id):
        try:
            result = self.db.conversations_collection.find_one({'user_id':user_id})
            if result is not None:
                result.pop('_id')
                return result
            else:
                return None
        except Exception as e:
            logger.error("mongo exception issue-{}".format(str(e)))        
            return None