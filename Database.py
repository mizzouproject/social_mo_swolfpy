import sqlite3
import os
from bw2data.project import projects

class DatabaseMultiOpt():
    def __init__(self):
        self.con = self._connect()
    
    def _create_db(self):
        cur = self.con.cursor()
        iteration_table = cur.execute("SELECT name FROM sqlite_master where name = 'iteration'")
        iteration_table_result = iteration_table.fetchall()
        if len(iteration_table_result) != 0:
            print("Iteration Table already present in the db, deleting...")
            cur.execute("DROP TABLE iteration;")    
        print("Creating temporal table to store iterations' information")
        cur.execute("CREATE TABLE iteration(" + 
                                "title TEXT, "+
                                "n_gen INTEGER, "+ 
                                "running_history TEXT, "+ 
                                "history_data TEXT);"
                    )
        print("Iteration Table was created successfully.!!!")
        
    def _connect(self):
        connection = sqlite3.connect(os.path.join(projects.dir, "lci", "temp.db"))
        return connection
        
    def _close_connection(self):
        self.con.close()
        
    def execute(self, query, return_data= False, parameters=None):
        cur = self.con.cursor()
        if parameters == None:
            res = cur.execute(query)
        else:
            cur.execute(query, parameters)
            self.con.commit()
        if return_data:
            return res.fetchall() 

    def insert_iteration(self, data):
        sql = "INSERT INTO iteration (title, n_gen, running_history, history_data) VALUES (?,?,?,?)"
        self.con.execute(sql, data)
        self.con.commit()

    def get_iterations(self):
        sql = "SELECT * FROM iteration"
        cur = self.con.cursor()
        res = cur.execute(sql)
        return res.fetchall() 