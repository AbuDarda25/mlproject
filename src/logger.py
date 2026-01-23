
# Logger is for the purpose that any execution that probably happens ,
#  we should be able to log all those information , the execution,
# everything in some files so that we'll be able to track 
# if there is some errors, even the exception error , we will try to
# log that into a text file
#  for that we need to implement that logger

import logging 
import os
from datetime import datetime 

# LOG_FILE = in this naming convention text file will be created
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True) # ok=True says that even though there is a File/Folder keep on appending the files inside that whenever we want to create the file.

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE) 

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s -%(levelname)s - %(message)s",
    level=logging.INFO,
    
)

