from diagrams import Cluster, Diagram
from diagrams.aws.storage import S3
from diagrams.custom import Custom  # Usamos Custom en lugar de LocalStorage
from diagrams.onprem.client import User
from diagrams.programming.language import Python

# Rutas a imágenes personalizadas
LOCAL_IMAGE = "./src/visualization/computer.png"
LOCAL_IMAGE2 = "./src/visualization/ml.png"
LOCAL_IMAGE3 = "./src/visualization/seaborn.png"
LOCAL_IMAGE4 = "./src/visualization/pandas.png"

with Diagram("Flujo proyecto de datos", show=True, direction="LR"):  # Corrección aquí
    # Representar la PC local como usuario
    user = User("Usuario Local")

    # Representar almacenamiento local con una imagen personalizada
    local_data = Custom("Datos en PC", LOCAL_IMAGE)

    python3 = Python("Python3 - Data Cleaning")
    python_3 = Python("Python3 - Data Load")

    # AWS Components
    raw_data = S3("S3 - Raw Data")
    processed_data = S3("S3 - Processed Data")

    #  Cluster con orientación vertical (solo afecta al contenido del Cluster)
    with Cluster("Data Driven"):
        driven = [
            Custom("ML - Train Data", LOCAL_IMAGE2),
            Custom("ML - Visualization", LOCAL_IMAGE3),
            Custom("ML - Analysis", LOCAL_IMAGE4),
        ]

    # Conectar elementos en el flujo
    user >> local_data >> python_3 >> raw_data >> python3 >> processed_data
    processed_data >> driven
