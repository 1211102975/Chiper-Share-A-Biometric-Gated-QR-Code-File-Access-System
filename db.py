import pyodbc

def get_db_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LAPTOP-V7J2F78U;"
        "DATABASE=fyp_database;"
        "Trusted_Connection=yes;"
    )
