"""
Utility functions for SQL queries and database connections to AACT database.
"""


import os
from dotenv import load_dotenv
load_dotenv()

#File specific imports
import pandas as pd
import psycopg
from sqlalchemy import create_engine

def connect_to_aact():
    """Create SQLAlchemy engine for AACT database using environment variables"""
    username = os.getenv('aact_username')
    password = os.getenv('aact_password')
    host = "aact-db.ctti-clinicaltrials.org"
    port = "5432"
    dbname = "aact"
    
    db_url = f"postgresql+psycopg://{username}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(db_url)
    return engine




def get_table(query):
    """
    Get table from AACT database using SQL query
    """
    
    conn = connect_to_aact()
    
    # Execute the SQL query and fetch the results into a DataFrame
    df = pd.read_sql(query, conn)
        
    return df