import os

from pathlib import Path
import requests
from tqdm import tqdm

# ImageNet WordNet ID --> image urls mappings download path
url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

# Data directory to save images of ImageNet
image_net_dir = os.path.join(str(Path(__file__).absolute().parent.parent), 'imageNet')
image_net_urls_file = os.path.join(image_net_dir, 'imagenet_fall11_urls.tgz')
image_net_image_dir = os.path.join(image_net_dir, 'images')

# Create these two repository if not exists
if not os.path.exists(image_net_dir):
    os.makedirs(image_net_dir)
if not os.path.exists(image_net_image_dir):
    os.makedirs(image_net_image_dir)


def download_img_urls():
    print('Downloading Image Net WID -> Image Urls Mappings file...')
    r = requests.get(url, stream=True)
    with open(image_net_urls_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        print('Total content length :', total_length)
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length/1024)+1, unit='KB'):
            f.write(chunk)
            f.flush()
    print('Download complete')


download_img_urls()
