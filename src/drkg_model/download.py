# @Author Arif YILMAZ, a.yilmaz@maastrichtuniversity.nl
# @repoaddress "https://github.com/arifx/XPREDICT"

import os

from openpredict.utils import get_openpredict_dir


# Downloading 500M kgpredict external dependency to avoid to have to commit it to dvc
# Maybe we can import the url in the dvc data folder? Like a pointer to the data
def download():
    if not os.path.exists(get_openpredict_dir("kgpredict/embed")):
        print("kgpredict data not present, downloading it")
        try:
            os.system(f'mkdir -p ./data/kgpredict')
            os.system(f"wget -q --show-progress purl.org/kgpredict -O kgpredict.tar.gz")
            os.system(f'tar -xzvf kgpredict.tar.gz  -C ./data/kgpredict/')
            os.system(f"rm kgpredict.tar.gz")
        except Exception as e:
            print(f"Error while downloading kgpredict: {e}")
