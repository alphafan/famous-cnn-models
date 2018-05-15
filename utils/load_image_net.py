import os
import tarfile
import multiprocessing as mp

from pathlib import Path
import requests
from tqdm import tqdm
import string
import random

# ImageNet WordNet ID --> image urls mappings download path
# ..
# n00004475_6590   http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg
# n00004475_15899  http://farm4.static.flickr.com/3276/2875184020_9944005d0d.jpg
# n00004475_32312  http://farm3.static.flickr.com/2531/4094333885_e8462a8338.jpg
# ..
wids_url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

# ImageNet WordNet ID --> object type mappings download path
# ..
# n00001740   entity
# n00001930   physical entity
# n00002137   abstraction, abstract entity
# ..
word_url = 'http://image-net.org/archive/words.txt'

# Data directory to save images of ImageNet
image_net_dir = os.path.join(str(Path(__file__).absolute().parent.parent), 'imageNet')
image_net_urls_file = os.path.join(image_net_dir, 'imagenet_fall11_urls.tgz')
image_net_word_file = os.path.join(image_net_dir, 'word.txt')
image_net_image_dir = os.path.join(image_net_dir, 'images')

# Create these two repository if not exists
if not os.path.exists(image_net_dir):
    os.makedirs(image_net_dir)
if not os.path.exists(image_net_image_dir):
    os.makedirs(image_net_image_dir)


def load_data(num_images=200, reshape_as=(227, 227)):
    """ Load ImageNet Dataset

    Args:
        num_images: ImageNet has 82,114 ids, for fast processing purpose,
            use only a portion of images for processing
         reshape_as: Pre-process image to make them all same width and height.
    """
    # Download files if not exists
    if not os.path.exists(image_net_urls_file) or os.path.getsize(image_net_urls_file) != 350302759:
        download_img_urls()
        assert os.path.getsize(image_net_urls_file) == 350302759
    if not os.path.exists(image_net_word_file) or os.path.getsize(image_net_word_file) != 2655750:
        download_word_file()
        assert os.path.getsize(image_net_word_file) == 2655750

    # Extract datasets
    print('Extracting urls from imagenet_fall11_urls.tgz...')
    url2name = {}
    with tarfile.open(image_net_urls_file) as tar:
        f = tar.extractfile('fall11_urls.txt')
        count = 0
        while count < num_images:
            filename, url = f.readline().decode().strip().split('\t')
            url2name[url] = filename
            count += 1

    # Download Images ...
    print('Parallel Downloading Images from Urls...')
    pool = mp.Pool(4)
    download_jobs = [pool.apply_async(download_image, (url, filename, )) for url, filename in url2name.items()]
    [job.get() for job in download_jobs]


def download_image(url, filename):
    try:
        print('Downloading', filename, 'from', url, '...')
        with open(os.path.join(image_net_image_dir, filename+'.jpg'), 'wb') as f:
            f.write(requests.get(url).content)
    except ConnectionError:
        print('Download Failed', url)


def download_img_urls():
    print('Downloading Image Net WID -> Image Urls Mappings file...')
    r = requests.get(wids_url, stream=True)
    with open(image_net_urls_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        print('Total content length :', total_length)
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1, unit='KB'):
            f.write(chunk)
            f.flush()


def download_word_file():
    print('Downloading Image Net WID -> Object Type Mappings file...')
    r = requests.get(word_url, stream=True)
    with open(image_net_word_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        print('Total content length :', total_length)
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1, unit='KB'):
            f.write(chunk)
            f.flush()


load_data()
