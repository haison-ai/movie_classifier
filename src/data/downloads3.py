import os

import boto3


class transform_data:
    def __init__(self, region: str = "us-east-2"):
        self.s3 = boto3.client("s3", region_name=region)

    def download_file(self, s3_name: str, local_path: str, s3_key: str):
        try:
            file_name = os.path.basename(s3_key)
            local_file_path = os.path.join(local_path, file_name)

            if os.path.exists(local_file_path):
                print(f"file does exist in: {local_file_path}")
                return local_file_path

            self.s3.download_file(s3_name, s3_key, local_file_path)
            print("files downloaded succesufully")
            return local_file_path

        except Exception as e:
            print(f"Error while downloading file: {e}")
            return None


if __name__ == "__main__":
    transform_data = transform_data()
    data = transform_data.download_file(
        "movieclassifiers3", "data/processed", "raw/ratings.dat"
    )
