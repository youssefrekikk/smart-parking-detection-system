# test_db_connection.py
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text

load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Admin123.")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "parking_detection")

print(f"Testing connection to: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Try with psycopg2 first
try:
    print("Testing with psycopg2...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"PostgreSQL version: {version[0]}")
    
    # List tables
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
    tables = cursor.fetchall()
    print("Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    conn.close()
    print("psycopg2 connection test: SUCCESS")
except Exception as e:
    print(f"psycopg2 connection test: FAILED - {str(e)}")

# Now try with SQLAlchemy
try:
    print("\nTesting with SQLAlchemy...")
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        version = result.scalar()
        print(f"PostgreSQL version: {version}")
        
        # List tables
        result = connection.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
        tables = result.fetchall()
        print("Tables in database:")
        for table in tables:
            print(f"  - {table[0]}")
    
    print("SQLAlchemy connection test: SUCCESS")
except Exception as e:
    print(f"SQLAlchemy connection test: FAILED - {str(e)}")
