import sqlite3
from utils import Utils

def test_fact_checker_performance():
    # Connect to the actual simulation database
    conn = sqlite3.connect('database/simulation.db')
    
    print("\nEvaluating fact checker performance on simulation data:")
    Utils.evaluate_fact_checker_performance(conn)
    
    conn.close()

if __name__ == "__main__":
    test_fact_checker_performance()