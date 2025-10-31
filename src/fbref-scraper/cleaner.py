import sqlite3
import pandas as pd
from datetime import datetime

from config import DBRAWDIR, DBRAWNAME, DBDIR, DBNAME, METADATATABLE, MATCHTABLE
from database_objects import DatabaseReader

def loadRawData(dbDir: str=DBRAWDIR, dbName: str=DBRAWNAME):
    with DatabaseReader(dbDir=dbDir, dbName=dbName) as db:
        metaDf = db.selectAll(tableName=METADATATABLE, orderby="id", asDf=True)
        statsDf = db.selectAll(tableName=MATCHTABLE, orderby="match_id", asDf=True)
    return metaDf, statsDf


def cleanFbrefData(rawDir: str=DBRAWDIR, rawName: str=DBRAWNAME, cleanDir: str=DBDIR, cleanName: str=DBNAME):
    metaDf, statsDf = loadRawData(rawDir, rawName)

