"""
Functions for interacting with S3 to save
images of the sumcurve within partition regions.
"""
import boto3
from botocore.exceptions import ClientError


def get_s3(aws_access_key_id=None, aws_secret_access_key=None):
    return boto3.resource(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )


def upload_file(file_name, bucket='voigt',
                object_name=None, aws_access_key_id=None,
                aws_secret_access_key=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    if aws_access_key_id is None or aws_secret_access_key is None:
        return False

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        return False
    return True
