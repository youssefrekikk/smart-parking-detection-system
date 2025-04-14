import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import logging
import psycopg2


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load environment variables from .env file
load_dotenv()

# Database connection settings from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Admin123.")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "parking_detection")

# Create SQLAlchemy database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# Log connection attempt (without password)
logger.info(f"Connecting to database: postgresql://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{DB_NAME}")
# Create SQLAlchemy engine with connection pooling
try:
    logger.info("Testing direct connection with psycopg2...")
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    conn.close()
    logger.info("Direct psycopg2 connection successful")
except Exception as e:
    logger.error(f"Direct psycopg2 connection failed: {str(e)}")
    # Continue anyway to see if SQLAlchemy can connect

# Now try with SQLAlchemy
try:
    # Create SQLAlchemy engine with connection pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=1800,
        echo=True,  # Set to True to log all SQL queries for debugging
    )
    
    # Test connection with a simple query
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        logger.info(f"SQLAlchemy test query result: {result.scalar()}")
        logger.info("Successfully connected to database with SQLAlchemy")
        
except Exception as e:
    logger.error(f"Failed to connect to database with SQLAlchemy: {str(e)}")
    # You could implement a fallback here if needed
    raise
# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all models
Base = declarative_base()

# Dependency to get DB session
def get_db():
    """
    Get a database session.
    This function is used as a FastAPI dependency to provide database sessions to endpoints.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
