"""
Database Explorer for Cycling Safety SVI Project

This script provides an overview of the SQLite database structure
and contents, displaying tables, their schemas, and sample data.
"""

import os
import sqlite3
import pandas as pd
import argparse
from pathlib import Path
import logging
from tabulate import tabulate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_database(db_path):
    """
    Connect to the SQLite database
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        sqlite3.Connection object
    """
    logger.info(f"Connecting to database: {db_path}")
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def get_table_list(conn):
    """
    Get a list of all tables in the database
    
    Args:
        conn: sqlite3.Connection object
        
    Returns:
        List of table names
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def get_table_schema(conn, table_name):
    """
    Get the schema for a specific table
    
    Args:
        conn: sqlite3.Connection object
        table_name: Name of the table
        
    Returns:
        List of tuples containing column information
    """
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()

def get_table_row_count(conn, table_name):
    """
    Get the number of rows in a table
    
    Args:
        conn: sqlite3.Connection object
        table_name: Name of the table
        
    Returns:
        Number of rows in the table
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    return cursor.fetchone()[0]

def get_sample_data(conn, table_name, limit=5):
    """
    Get sample data from a table
    
    Args:
        conn: sqlite3.Connection object
        table_name: Name of the table
        limit: Maximum number of rows to retrieve
        
    Returns:
        pandas DataFrame containing sample data
    """
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        return pd.read_sql_query(query, conn)
    except pd.errors.DatabaseError as e:
        logger.error(f"Error reading data from table {table_name}: {e}")
        return pd.DataFrame()

def format_table_schema(schema):
    """
    Format table schema for display
    
    Args:
        schema: List of tuples containing column information
        
    Returns:
        Formatted schema as a list of dictionaries
    """
    formatted_schema = []
    for col in schema:
        formatted_schema.append({
            'cid': col[0],
            'name': col[1],
            'type': col[2],
            'notnull': 'NOT NULL' if col[3] else '',
            'default': col[4] if col[4] is not None else '',
            'pk': 'PRIMARY KEY' if col[5] else ''
        })
    return formatted_schema

def display_database_overview(db_path):
    """
    Display an overview of the database structure and contents
    
    Args:
        db_path: Path to the SQLite database file
    """
    # Connect to the database
    conn = connect_to_database(db_path)
    
    # Get list of tables
    tables = get_table_list(conn)
    logger.info(f"Found {len(tables)} tables in the database")
    
    # Print overall database info
    print("\n===== DATABASE OVERVIEW =====")
    print(f"Database: {os.path.basename(db_path)}")
    print(f"Tables: {len(tables)}")
    print("=" * 30)
    
    # Display summary of tables
    summary_data = []
    for table in tables:
        row_count = get_table_row_count(conn, table)
        schema = get_table_schema(conn, table)
        col_count = len(schema)
        summary_data.append([table, col_count, row_count])
    
    print("\n===== TABLE SUMMARY =====")
    print(tabulate(summary_data, headers=["Table Name", "Columns", "Rows"], tablefmt="grid"))
    
    # For each table, display schema and sample data
    for table in tables:
        print(f"\n\n===== TABLE: {table} =====")
        
        # Get and display schema
        schema = get_table_schema(conn, table)
        formatted_schema = format_table_schema(schema)
        print("\nSchema:")
        print(tabulate(formatted_schema, headers="keys", tablefmt="grid"))
        
        # Get and display sample data
        sample_data = get_sample_data(conn, table)
        if not sample_data.empty:
            print("\nSample Data (first 5 rows):")
            print(tabulate(sample_data, headers="keys", tablefmt="grid", showindex=False))
        else:
            print("\nNo sample data available or table is empty.")
    
    # Close the connection
    conn.close()

def analyze_relationships(db_path):
    """
    Analyze potential relationships between tables
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Dictionary of potential foreign key relationships
    """
    conn = connect_to_database(db_path)
    tables = get_table_list(conn)
    
    # Get all column names for each table
    table_columns = {}
    for table in tables:
        schema = get_table_schema(conn, table)
        table_columns[table] = [col[1] for col in schema]
    
    # Look for potential relationships
    relationships = []
    
    for table1 in tables:
        for table2 in tables:
            if table1 != table2:
                for col1 in table_columns[table1]:
                    # Common patterns for foreign keys
                    if col1 in table_columns[table2] or f"{table2}_id" == col1 or f"{col1}_id" == table2:
                        # Check if the column values in table1 are contained in table2
                        try:
                            query = f"""
                            SELECT COUNT(*) FROM {table1} t1
                            LEFT JOIN {table2} t2 ON t1.{col1} = t2.{col1 if col1 in table_columns[table2] else 'id'}
                            WHERE t2.{col1 if col1 in table_columns[table2] else 'id'} IS NULL
                            AND t1.{col1} IS NOT NULL
                            """
                            cursor = conn.cursor()
                            cursor.execute(query)
                            non_matching = cursor.fetchone()[0]
                            
                            # If most values match, it's likely a relationship
                            total_query = f"SELECT COUNT(*) FROM {table1} WHERE {col1} IS NOT NULL"
                            cursor.execute(total_query)
                            total = cursor.fetchone()[0]
                            
                            if total > 0 and non_matching / total < 0.1:  # Less than 10% non-matching values
                                relationships.append({
                                    'table1': table1, 
                                    'column1': col1, 
                                    'table2': table2, 
                                    'column2': col1 if col1 in table_columns[table2] else 'id',
                                    'confidence': (total - non_matching) / total if total > 0 else 0
                                })
                        except sqlite3.Error:
                            # Skip if the query fails
                            pass
    
    # Sort relationships by confidence
    relationships.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Close the connection
    conn.close()
    
    return relationships

def main():
    """Main function to parse arguments and run the database exploration"""
    parser = argparse.ArgumentParser(description='Explore and display database structure')
    
    parser.add_argument('--db-path', type=str, default='data/raw/database_2024_10_07_135133.db',
                        help='Path to SQLite database file')
    parser.add_argument('--analyze-relationships', action='store_true',
                        help='Analyze potential relationships between tables')
    
    args = parser.parse_args()
    
    # Display database overview
    display_database_overview(args.db_path)
    
    # Analyze relationships if requested
    if args.analyze_relationships:
        print("\n\n===== POTENTIAL TABLE RELATIONSHIPS =====")
        relationships = analyze_relationships(args.db_path)
        
        if relationships:
            print(tabulate(relationships, headers="keys", tablefmt="grid"))
        else:
            print("No potential relationships detected.")

if __name__ == "__main__":
    main() 