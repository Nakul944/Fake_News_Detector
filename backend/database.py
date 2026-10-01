import os

import mysql.connector
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Database configuration
# ---------------------------------------------------------

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "fake_news_detector")


# ---------------------------------------------------------
# Get database connection
# ---------------------------------------------------------

def get_db_connection():

    connection = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    return connection


# ---------------------------------------------------------
# Save prediction
# ---------------------------------------------------------

def save_prediction(
    claim: str,
    predicted_label: str,
    confidence: float
):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        query = """
        INSERT INTO prediction_history
        (
            claim,
            predicted_label,
            confidence
        )
        VALUES (%s, %s, %s)
        """

        values = (
            claim,
            predicted_label,
            confidence
        )

        cursor.execute(query, values)

        connection.commit()

        return cursor.lastrowid

    except Exception as e:

        print(f"Database error: {e}")

        # We don't want the whole prediction API
        # to crash just because database saving failed.
        return None

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()


# ---------------------------------------------------------
# Fetch recent predictions
# ---------------------------------------------------------

def get_prediction_history(limit: int = 20):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()

        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            id,
            claim,
            predicted_label,
            confidence,
            created_at
        FROM prediction_history
        ORDER BY created_at DESC
        LIMIT %s
        """

        cursor.execute(query, (limit,))

        results = cursor.fetchall()

        return results

    except Exception as e:

        print(f"Database error: {e}")

        return []

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()