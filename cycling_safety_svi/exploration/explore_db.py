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

def get_demographic_mappings():
    """
    Returns the demographic mappings dictionary.
    
    This is copied from SafetyDemographicsInteractionModel to avoid instantiating the class.
    """
    return {
        'age': {
            1: '18-30 years', 2: '31-45 years', 3: '46-60 years', 
            4: '61-75 years', 5: '76+ years'
        },
        'gender': {
            1: 'Male', 2: 'Female', 3: 'Other', 4: 'Prefer not to say'
        },
        'household_composition': {
            1: 'Live alone', 2: 'Couple without children', 3: 'Couple with children',
            4: 'One adult with children', 5: 'Two or more adults (not couple)', 6: 'Other'
        },
        'household_size': {
            1: '1 person', 2: '2 people', 3: '3 people', 
            4: '4 people', 5: '5 people', 6: '6+ people'
        },
        'education': {
            1: 'No education', 2: 'Primary education', 3: 'Lower vocational',
            4: 'Lower secondary', 5: 'Intermediate vocational', 6: 'MULO or MMS',
            7: 'HAVO', 8: 'HBS, VWO, etc.', 9: 'Higher vocational (HBO)',
            10: 'University', 11: 'M.Sc.', 12: 'Ph.D.',
            13: 'Other', 14: 'Prefer not to say'
        },
        'income': {
            1: '< €1,250', 2: '€1,251-€1,700', 3: '€1,701-€2,250',
            4: '€2,251-€3,650', 5: '€3,651-€7,000', 6: '> €7,001',
            7: 'Unknown', 8: 'Prefer not to say'
        },
        'bills': {
            1: 'Very easy', 2: 'Easy', 3: 'Reasonable', 
            4: 'Difficult', 5: 'Very difficult', 6: 'Unknown'
        },
        'transportation': {
            1: 'Walking', 2: 'Bike', 3: 'Public transport', 4: 'Car', 5: 'Other'
        },
        'car': {
            1: 'No cars', 2: '1 car', 3: '2 cars', 4: '3+ cars'
        },
        'traveltime': {
            1: 'No commute', 2: '< 10 min', 3: '10-20 min',
            4: '20-30 min', 5: '30-40 min'
        },
        'commutingdays': {
            1: 'No commute', 2: '1 day/week', 3: '2 days/week',
            4: '3 days/week', 5: '4 days/week', 6: '5+ days/week'
        },
        'cycler': {
            1: 'Do not cycle', 2: '< 1/week', 3: '1 day/week',
            4: '2 days/week', 5: '3 days/week', 6: '4 days/week',
            7: '5+ days/week'
        },
        'cyclingincident': {
            1: 'Yes, severe', 2: 'Yes, mild', 3: 'No'
        },
        'cyclinglike': {1: 'Yes', 2: 'No'},
        'cyclingunsafe': {
            1: 'Yes, sometimes', 2: 'Yes, evening/night', 3: 'No'
        },
        'biketype': {
            1: 'Regular bike', 2: 'Racing bike', 3: 'E-bike', 
            4: 'Fatbike', 5: 'Other'
        },
        'trippurpose': {
            1: 'Commuting', 2: 'Errands', 3: 'Recreational', 4: 'Other'
        }
    }

def save_response_data_as_latex(db_path, output_dir='reports/models'):
    """
    Saves the first 5 rows of the Response table as a LaTeX file.

    This function selects only the columns relevant to the safety-demographics
    interaction model, formats them, and saves them as a .tex file.

    Args:
        db_path (str): Path to the SQLite database.
        output_dir (str): Directory to save the output file.
    """
    logger.info("Generating LaTeX sample table from Response data...")
    conn = connect_to_database(db_path)
    
    demographic_mappings = get_demographic_mappings()
    
    # Columns used in safety_demographics_interaction_model.py plus response columns
    relevant_columns = list(demographic_mappings.keys())
    if 'work' in relevant_columns:
        relevant_columns.remove('work')

    query_columns = relevant_columns + ['resp_main_1', 'resp_main_15']

    try:
        query = f"SELECT {', '.join(query_columns)} FROM Response LIMIT 5"
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        logger.error(f"Failed to query Response table: {e}")
        return
    finally:
        conn.close()

    # Drop rows with NaN before processing
    df.dropna(inplace=True)

    # Map integer codes to string categories
    for col, mapping in demographic_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(int).map(mapping)

    # Add '...' column and reorder
    if 'resp_main_15' in df.columns:
        df.insert(df.columns.get_loc('resp_main_15'), '...', '...')

    # Prettify column names
    df.columns = [col.replace('_', ' ').title() for col in df.columns]

    # Generate and save LaTeX table
    output_path = Path(output_dir) / 'response_data_sample.tex'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latex_string = df.to_latex(
        index=False,
        caption='Sample of Demographics Data from the Response Table.',
        label='tab:demographics_sample',
        position='htbp',
        longtable=True,
        escape=False
    )
    
    # Use resizebox for wide tables
    if len(df.columns) > 10:
        latex_string = latex_string.replace('\\begin{longtable}', '\\resizebox{\\textwidth}{!}{\\begin{tabular}')
        latex_string = latex_string.replace('\\end{longtable}', '\\end{tabular}}')
        latex_string = latex_string.replace('{longtable}', '{tabular}')
    
    with open(output_path, 'w') as f:
        f.write(latex_string)
    
    logger.info(f"Successfully saved sample response data to {output_path}")

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
    parser.add_argument('--save-latex', action='store_true', default=True,
                        help='Save a LaTeX sample of the Response table')
    
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

    # Save LaTeX table if requested
    if args.save_latex:
        save_response_data_as_latex(args.db_path)

if __name__ == "__main__":
    main() 