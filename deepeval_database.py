# This code is used to create a database , insert the data from the results of unit testcases.

import psycopg2
import json
import os
from dotenv import load_dotenv

load_dotenv()

directory_path = os.environ['DEEPEVAL_RESULTS_FOLDER']

class DeepEvalDatabase:
    def __init__(self, hostname, database, username, password, port):
        self.hostname = hostname
        self.database = database
        self.username = username
        self.password = password
        self.port = port

    def connection(self):
        try:
            self.conn = psycopg2.connect(
                host=self.hostname,
                dbname=self.database,
                user=self.username,
                password=self.password,
                port=self.port,
            )
            self.cur = self.conn.cursor()
            print("Connected to the database")
        except psycopg2.Error as e:
            print("Error in connecting:", e)
            
    def rename_files_to_json(self,directory):
        renamed_files = []
        for filename in os.listdir(directory):
            if not filename.endswith(".json"):
                new_filename = os.path.splitext(filename)[0] + ".json"
                try:
                    os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                    renamed_files.append(new_filename) 
                except OSError as e:
                    print(f"Error renaming {filename}: {e}")
        return renamed_files

    def insert_data(self, json_file_path):
        with open(json_file_path, "r") as file:
            json_data = json.load(file)
            
            testfile = json_data["testFile"]
            deployment = json_data["deployment"]
            testcase = json.dumps(json_data["testCases"])
            
            insert_statements = """
                INSERT INTO test_schema.deepeval(testfile, deployment, testcase)
                VALUES (%s,%s,%s);
            """
            
            try:
                self.cur.execute(insert_statements, (testfile, deployment, testcase))
                self.conn.commit()
                print(f"Data Inserted Successfully from: {json_file_path}")
            except psycopg2.Error as e:
                print(f"Error in inserting data from {json_file_path}:", e)

if __name__ == "__main__":
    hostname = os.environ['HOSTNAME']
    database = os.environ['DATABASE']
    username = os.environ['USERNAME']
    password = os.environ['PASSWORD']
    port = os.environ['PORT']

    deepeval_db = DeepEvalDatabase(hostname, database, username, password, port)
    deepeval_db.connection()

    renamed_filenames = deepeval_db.rename_files_to_json(directory_path)
    for filename in renamed_filenames:
        json_file_path = os.path.join(directory_path, filename)
        deepeval_db.insert_data(json_file_path)

