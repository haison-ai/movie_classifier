import os
import boto3


class LoadDatas3:
    def __init__(self, s3_name: str, s3_folder: str, region: str = "us-east-2"):
        """
        With this class gonna to upload the raw data in s3 bucket.
        """
        self.s3_name = s3_name
        self.s3_folder = s3_folder
        self.s3 = boto3.client("s3", region_name=region)

    def load_s3(self, local_path: str):
        """
        Method to load the s3 bucket.
        """
        try:
            if not os.path.exists(local_path):
                print(f" file doesn't exist in: {local_path}")
                return False
            # cleating client in s3
            file_name = os.path.basename(local_path)
            s3_path = f"{self.s3_folder}/{file_name}"

            self.s3.upload_file(local_path, self.s3_name, s3_path)
            print(
                f" files uploaded succesufully: {local_path} → s3://{self.s3_name}/{s3_path}"
            )
            return s3_path

        except Exception as e:
            print(f" Error to upload file: {e}")
            return False


if __name__ == "__main__":
    upload = LoadDatas3("movieclassifiers3", "raw")
    upload.load_s3("data/raw/users.dat")
