from sdv.datasets.demo import get_available_demos
from sdv.datasets.demo import download_demo
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
import os
import pandas as pd

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CURL_CA_BUNDLE']=os.path.join(os.getcwd(), "root.pem")
os.environ['SSL_CERT_FILE']=os.path.join(os.getcwd(), "root.pem")
os.environ['REQUESTS_CA_BUNDLE']=os.path.join(os.getcwd(), "root.pem")
os.environ['NODE_EXTRA_CA_CERTS']=os.path.join(os.getcwd(), "root.pem")

# print(get_available_demos(modality='single_table'))
data, metadata = download_demo(modality = 'single_table',
                               dataset_name = 'fake_hotel_guests')

print(metadata)


# data.to_csv('datasets/test_data.csv', index=False, encoding='utf-32')

# import chardet

# with open('datasets/test_data.csv', "rb") as f:
#     result = chardet.detect(f.read())
# print(result)

datasets = load_csvs(
    folder_name = 'datasets/',
    read_csv_parameters={
        'skipinitialspace': True,
        'encoding': 'utf_32'
    }
)

# # data = datasets['guests']
# print(datasets["test_data"])

meta = Metadata.detect_from_dataframe(
    data=datasets["test_data"],
    table_name='hotel_people'
)

print(meta)