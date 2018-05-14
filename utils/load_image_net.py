import os

from pathlib import Path
import requests
from clint.textui import progress
from tqdm import tqdm

# ImageNet WordNet ID --> image urls mappings download path
url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

# Data directory to save images of ImageNet
imageNetDir = os.path.join(str(Path(__file__).absolute().parent.parent), 'imageNet')
imageNetUrlsFile = os.path.join(imageNetDir, 'imagenet_fall11_urls.tgz')
imageNetImageDir = os.path.join(imageNetDir, 'images')

# Create these two repository if not exists
if not os.path.exists(imageNetDir):
    os.makedirs(imageNetDir)
if not os.path.exists(imageNetImageDir):
    os.makedirs(imageNetImageDir)


def download_img_urls():
    """ Download from ImageNet website the WordNet id --> image url mappings. """
    print('Downloading Image Net WID -> Image Urls Mappings file...')
    r = requests.get(url, stream=True)
    with open(imageNetUrlsFile, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        print('Total content length :', total_length)
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length/1024)+1, unit='KB'):
            f.write(chunk)
            f.flush()
    print('Download complete')


download_img_urls()
