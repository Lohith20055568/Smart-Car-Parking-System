import copy
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from .config import MONGO_URI, DATABASE_NAME

_client = None
_db = None
_memory = {
    'slots': [],
    'detections': [],
    'events': []
}

def clean_doc(doc):
    """
    Recursively sweeps any dictionary, list, or nested structure 
    and removes/converts any PyMongo ObjectId into standard JSON-serializable types.
    """
    if isinstance(doc, list):
        return [clean_doc(item) for item in doc]
    if isinstance(doc, dict):
        res = {}
        for k, v in doc.items():
            if k == '_id':
                continue
            if isinstance(v, ObjectId):
                res[k] = str(v)
            else:
                res[k] = clean_doc(v)
        return res
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

def get_db():
    global _client, _db
    if _db is not None:
        return _db
    try:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2500)
        _client.admin.command('ping')
        _db = _client[DATABASE_NAME]
        _db.slots.create_index([('slot_id', ASCENDING)], unique=True)
        _db.detections.create_index([('created_at', ASCENDING)])
        return _db
    except (PyMongoError, ServerSelectionTimeoutError):
        return None

def using_memory():
    return get_db() is None

def seed_slots():
    default_slots = [
        {'slot_id': 'A1', 'x1': 60, 'y1': 130, 'x2': 190, 'y2': 300, 'status': 'unknown'},
        {'slot_id': 'A2', 'x1': 210, 'y1': 130, 'x2': 340, 'y2': 300, 'status': 'unknown'},
        {'slot_id': 'A3', 'x1': 360, 'y1': 130, 'x2': 490, 'y2': 300, 'status': 'unknown'},
        {'slot_id': 'B1', 'x1': 60, 'y1': 330, 'x2': 190, 'y2': 500, 'status': 'unknown'},
        {'slot_id': 'B2', 'x1': 210, 'y1': 330, 'x2': 340, 'y2': 500, 'status': 'unknown'},
        {'slot_id': 'B3', 'x1': 360, 'y1': 330, 'x2': 490, 'y2': 500, 'status': 'unknown'},
    ]
    db = get_db()
    if db is None:
        if not _memory['slots']:
            _memory['slots'] = default_slots
        return default_slots
    if db.slots.count_documents({}) == 0:
        db.slots.insert_many(copy.deepcopy(default_slots))
    return clean_doc(list(db.slots.find({}, {'_id': 0})))

def get_slots():
    db = get_db()
    if db is None:
        return _memory['slots'] or seed_slots()
    return clean_doc(list(db.slots.find({}, {'_id': 0})))

def upsert_slots(slots):
    db = get_db()
    for s in slots:
        s['updated_at'] = datetime.utcnow()
            
    clean_slots = clean_doc(slots)
    
    if db is None:
        _memory['slots'] = clean_slots
        return clean_slots
        
    db.slots.delete_many({})
    if clean_slots:
        # Pass a deep copy so PyMongo doesn't mutate our local dictionaries
        db.slots.insert_many(copy.deepcopy(clean_slots))
        
    return clean_doc(list(db.slots.find({}, {'_id': 0})))

def save_detection(record):
    record['created_at'] = datetime.utcnow()
    db = get_db()
    
    if db is None:
        record['id'] = str(len(_memory['detections']) + 1)
        cleaned_record = clean_doc(record)
        _memory['detections'].append(cleaned_record)
        return cleaned_record

    # Pass a deep copy to PyMongo to protect the original 'record' from being mutated with _id
    record_to_insert = copy.deepcopy(record)
    result = db.detections.insert_one(record_to_insert)
    
    record['id'] = str(result.inserted_id)
    return clean_doc(record)

def latest_detections(limit=20):
    db = get_db()
    if db is None:
        return clean_doc(list(reversed(_memory['detections'][-limit:])))
    rows = list(db.detections.find({}, {'_id': 0}).sort('created_at', -1).limit(limit))
    return clean_doc(rows)
