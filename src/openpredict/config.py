import os

from pydantic import BaseSettings


class Settings(BaseSettings):

    PROD_URL: str = 'https://openpredict.transltr.io'
    TEST_URL: str = 'https://openpredict.test.transltr.io'
    STAGING_URL: str = 'https://openpredict.ci.transltr.io'
    DEV_URL: str = 'https://openpredict.semanticscience.org'

    VIRTUAL_HOST: str = None
    DEV_MODE: bool = False

    BIOLINK_VERSION: str = '2.3.0'
    TRAPI_VERSION: str = "1.3.0"

    OPENPREDICT_DATA_DIR: str = os.path.join(os.getcwd(), 'data')

settings = Settings()
