import copy
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from .config import MONGO_URI, DATABASE_NAME

_client = None
_db = None

_memory = {
    "slots": [],
    "detections": [],
    "events": []
}


def clean_doc(doc):
    if isinstance(doc, list):
        return [clean_doc(x) for x in doc]

    if isinstance(doc, dict):
        return {
            k: clean_doc(v)
            for k, v in doc.items()
            if k != "_id"
        }

    if isinstance(doc, ObjectId):
        return str(doc)

    return doc


def get_db():
    global _client, _db

    if _db is not None:
        return _db

    try:
        _client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000
        )

        _client.admin.command("ping")

        _db = _client[DATABASE_NAME]

        _db.slots.create_index(
            [("slot_id", ASCENDING)],
            unique=True
        )

        _db.detections.create_index(
            [("created_at", ASCENDING)]
        )

        print("MongoDB connected successfully")

        return _db

    except PyMongoError as e:
        print("MongoDB connection failed:", e)
        return None


def using_memory():
    return get_db() is None


def seed_slots():
    slots = [
        {"slot_id": "A1", "x1": 60, "y1": 130, "x2": 190, "y2": 300, "status": "unknown"},
        {"slot_id": "A2", "x1": 210, "y1": 130, "x2": 340, "y2": 300, "status": "unknown"},
        {"slot_id": "A3", "x1": 360, "y1": 130, "x2": 490, "y2": 300, "status": "unknown"},
        {"slot_id": "B1", "x1": 60, "y1": 330, "x2": 190, "y2": 500, "status": "unknown"},
        {"slot_id": "B2", "x1": 210, "y1": 330, "x2": 340, "y2": 500, "status": "unknown"},
        {"slot_id": "B3", "x1": 360, "y1": 330, "x2": 490, "y2": 500, "status": "unknown"}
    ]

    db = get_db()

    if db is None:
        if not _memory["slots"]:
            _memory["slots"] = slots
        return slots

    if db.slots.count_documents({}) == 0:
        db.slots.insert_many(copy.deepcopy(slots))

    return clean_doc(list(db.slots.find({}, {"_id": 0})))


def get_slots():
    db = get_db()

    if db is None:
        return _memory["slots"] or seed_slots()

    return clean_doc(
        list(db.slots.find({}, {"_id": 0}))
    )


def upsert_slots(slots):
    db = get_db()

    for slot in slots:
        slot["updated_at"] = datetime.utcnow()

    slots = clean_doc(slots)

    if db is None:
        _memory["slots"] = slots
        return slots

    db.slots.delete_many({})

    if slots:
        db.slots.insert_many(copy.deepcopy(slots))

    return clean_doc(
        list(db.slots.find({}, {"_id": 0}))
    )


def save_detection(record):
    record["created_at"] = datetime.utcnow()

    db = get_db()

    if db is None:
        record["id"] = str(
            len(_memory["detections"]) + 1
        )

        record = clean_doc(record)
        _memory["detections"].append(record)

        return record

    result = db.detections.insert_one(
        copy.deepcopy(record)
    )

    record["id"] = str(result.inserted_id)

    return clean_doc(record)


def latest_detections(limit=20):
    db = get_db()

    if db is None:
        return clean_doc(
            list(
                reversed(
                    _memory["detections"][-limit:]
                )
            )
        )

    rows = list(
        db.detections
        .find({}, {"_id": 0})
        .sort("created_at", -1)
        .limit(limit)
    )

    return clean_doc(rows)
