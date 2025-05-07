import os
import logging
import boto3
import botocore.exceptions


class DownloadDatas3:
    """Clase para descrgar los datos de s3"""

    def __init__(self, region: str = "us-east-2"):
        #Start client with aws s3
        self.region = region
        self.s3 = boto3.client("s3", region_name=self.region)

    def download_file(self, s3_name: str, local_path: str, s3_key: str) -> None:
        """Este metodo descarga los datos desde s3"""
        try:
            file_name = os.path.basename(s3_key)
            local_file_path = os.path.join(local_path, file_name)

            if os.path.exists(local_file_path):
                logging.info(f"file does exist in: {local_file_path}")
                return local_file_path

            self.s3.download_file(s3_name, s3_key, local_file_path)
            logging.info("files downloaded succesufully")
            return local_file_path

        except botocore.exceptions.NoCredentialsError:
            logging.error("AWS credentials not found. Run `aws configure` to set them.")
        except botocore.exceptions.ClientError as e:
            logging.error(f"S3 Client Error: {e}")
        except FileNotFoundError:
            logging.error(f"Directory not found: {local_path}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

        return None
