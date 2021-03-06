import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datajoint as dj
import os
dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, 'parameters.yaml')
with open(file_path) as file:
    parameters = yaml.load(file, Loader=yaml.FullLoader)

user = parameters['user']
password = parameters['password']
host = parameters['host']
database = parameters['schema']
connection_string = 'mysql+pymysql://{}:{}@{}/{}'.format(user, password, host, database)
engine = create_engine(connection_string, echo=False)
DBSession = sessionmaker(bind=engine)
session = DBSession()

##### DJ parameters
# Connect to the datajoint database
dj.config['database.user'] = user
dj.config['database.password'] = password
dj.config['database.host'] = host
dj.conn()


