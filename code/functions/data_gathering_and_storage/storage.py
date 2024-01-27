from pymongo.mongo_client import MongoClient
import pandas as pd

uri = "mongodb+srv://yuanyuan:Fh5siW6hR5Xk00vf@cluster0.jawthi4.mongodb.net/?retryWrites=true&w=majority"

def connect_to_mongoDB():
    # Create a new client and connect to the server
    client = MongoClient(uri)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)
        return None
    

def insert_data(data, db_name, collection_name):
    try:    
        client = MongoClient(uri)
        
        db = client[db_name] #database
        collection = db[collection_name] #collection
        
        # insert data
        data_records = data.to_dict(orient='records')
        result = collection.insert_many(data_records)
        
        print(f"Inserted {len(result.inserted_ids)} records into {db_name}.{collection_name}")
        
        return result
    except Exception as e:
        print(e)
        return None


def fetch_data_from_db(db_name, collection_name):
    # Create a new client and connect to the server
    client = MongoClient(uri)

    # Send a ping to confirm a successful connection
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
        return None

    # Access the specified database and collection
    db = client[db_name]
    collection = db[collection_name]

    # Fetch data from MongoDB
    data = list(collection.find())

    # Create a DataFrame
    df = pd.DataFrame(data)

    return df