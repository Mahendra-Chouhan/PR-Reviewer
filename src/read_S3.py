import boto3

print("started")

s3 = boto3.resource('s3',region_name='region_name', aws_access_key_id='your_access_id', aws_secret_access_key='your access key')

obj = s3.Object('bucket_name','file_name')

data=obj.get()['Body'].read()

print(data)
