import os
from flask import Flask, Response, request, jsonify, make_response
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.json_util import dumps
from bson.objectid import ObjectId

load_dotenv()

app = Flask(__name__)
mongo_db_url = os.environ.get("MONGO_DB_CONN_STRING")

client = MongoClient(mongo_db_url)
db = client['AAPLdb']

#Create
@app.post("/api/StockData")
def add_StockData():
    _json = request.json
    db.StockData.insert_one(_json)

    resp = jsonify({"message": "StockData added successfully"})
    resp.status_code = 200
    return resp

#Read
@app.get("/api/StockData")
def get_StockData():
    Date = request.args.get('Date')
    filter = {} if Date is None else {"Date": Date}
    StockData = list(db.StockData.find(filter))

    response = Response(
        response=dumps(StockData), status=200,  mimetype="application/json")
    return response

#Update
@app.put("/api/StockData/<id>")
def update_StockData(id):
    _json = request.json
    db.StockData.update_one({'_id': ObjectId(id)}, {"$set": _json})

    resp = jsonify({"message": "StockData updated successfully"})
    resp.status_code = 200
    return resp

#Delete
@app.delete("/api/StockData/<id>")
def delete_StockData(id):
    db.StockData.delete_one({'_id': ObjectId(id)})

    resp = jsonify({"message": "StockData deleted successfully"})
    resp.status_code = 200
    return resp 


@app.errorhandler(400)
def handle_400_error(error):
    return make_response(jsonify({"errorCode": error.code, 
                                  "errorDescription": "Bad request!",
                                  "errorDetailedDescription": error.description,
                                  "errorName": error.name}), 400)

@app.errorhandler(404)
def handle_404_error(error):
        return make_response(jsonify({"errorCode": error.code, 
                                  "errorDescription": "Resource not found!",
                                  "errorDetailedDescription": error.description,
                                  "errorName": error.name}), 404)

@app.errorhandler(500)
def handle_500_error(error):
        return make_response(jsonify({"errorCode": error.code, 
                                  "errorDescription": "Internal Server Error",
                                  "errorDetailedDescription": error.description,
                                  "errorName": error.name}), 500)


if __name__ == '__main__':
     app.run()