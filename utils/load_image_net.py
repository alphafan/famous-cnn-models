import os
import tarfile

import cv2
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
from PIL import Image
import concurrent.futures

##########################################################################
# Two Files are given by ImageNet Website to download
#   - 1. A file containing Image WID --> image urls mappings
#   .. Example :
#   n00004475_6590   http://farm4.static.flickr.com/3175/2737866473_7958dc8760.jpg
#   n00004475_15899  http://farm4.static.flickr.com/3276/2875184020_9944005d0d.jpg
#   n00004475_32312  http://farm3.static.flickr.com/2531/4094333885_e8462a8338.jpg
#   ..
#   - 2. A file containing Image WID --> object type mappings
#   .. Example :
#   n00001740   entity
#   n00001930   physical entity
#   n00002137   abstraction, abstract entity
#   ..
##########################################################################

# ImageNet WordNet ID --> image urls mappings download path
wids_url = 'http://image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz'

# ImageNet WordNet ID --> object type mappings download path
word_url = 'http://image-net.org/archive/words.txt'

##########################################################################
# 1. Make directory to save download datasets
##########################################################################


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


##########################################################################
# 2. Download two files given by ImageNet
##########################################################################


def download_wid2urls_file():
    print('Downloading Image Net WID -> Image Urls Mappings file...')
    r = requests.get(wids_url, stream=True)
    with open(image_net_urls_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1, unit='KB'):
            f.write(chunk)
            f.flush()


# Download first file
if not os.path.exists(image_net_urls_file) or os.path.getsize(image_net_urls_file) != 350302759:
    download_wid2urls_file()
    assert os.path.getsize(image_net_urls_file) == 350302759


def download_wid2types_file():
    print('Downloading Image Net WID -> Object Type Mappings file...')
    r = requests.get(word_url, stream=True)
    with open(image_net_word_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1, unit='KB'):
            f.write(chunk)
            f.flush()


# Download second file
if not os.path.exists(image_net_word_file) or os.path.getsize(image_net_word_file) != 2655750:
    download_wid2types_file()
    assert os.path.getsize(image_net_word_file) == 2655750

print('Size of wid -> urls file : {:8.2f} Mb'.format(os.path.getsize(image_net_urls_file) / 1024 / 1024))
print('Size of wid -> word file : {:8.2f} Mb'.format(os.path.getsize(image_net_word_file) / 1024 / 1024))


##########################################################################
# 3. Extract urls from download zipped file
##########################################################################


def get_url2wid():
    """ Extract imagenet_fall11_urls.tgz file and get image download urls. """
    print('Extracting urls from imagenet_fall11_urls.tgz...')
    url2wid = {}
    with tarfile.open(image_net_urls_file) as tar:
        f = tar.extractfile('fall11_urls.txt')
        while True:
            try:
                wid, url = f.readline().decode().strip().split('\t')
                url2wid[url] = wid
            except ValueError:
                break
    return url2wid


# Url -> file name mapping
url2wid = get_url2wid()
print('Extract {} images download urls in total'.format(len(url2wid)))


##########################################################################
# 4. Download Images
##########################################################################


def can_open_image(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True


def is_image_valid(path):
    """
    Check if image is still valid
    No longer valid image's four corner are the same pixels
    """
    im = Image.open(path).convert('RGB')
    w, h = im.size
    if im.getpixel((0, 0)) == im.getpixel((w-1, 0)) == im.getpixel((0, h-1)) == im.getpixel((w-1, h-1)):
        return False
    return True


def download(url, wid):
    """ Download image and save it. """
    try:
        fmt = url.split('.')[-1].strip().lower()
        if fmt in ['jpg', 'png']:
            print('Downloading', wid, 'from', url, '...')
            r = requests.get(url)
            filepath = os.path.join(image_net_image_dir, wid + '.' + fmt)
            with open(filepath, 'wb') as f:
                f.write(r.content)
            # Filter image file that can not open
            # Filter image file that is not valid
            # Not valid image is the file where image no longer exists
            if not can_open_image(filepath) or not is_image_valid(filepath):
                os.remove(filepath)
    except requests.exceptions.RequestException:
        print('Failed to download', url)


def download_all():
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Start the load operations and mark each future with its URL
        futures = {executor.submit(download, url, wid)
                   for url, wid in list(url2wid.items())[:15000]}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print('Skip...', str(e))


if len(os.listdir(image_net_image_dir)) < 7000:
    download_all()

print('There are {} images downloaded.'.format(len(os.listdir(image_net_image_dir))))


##########################################################################
# 5. Reshape Images into Desired Size
##########################################################################

def load_image(filename):
    # Read image from file
    img = cv2.imread(os.path.join(image_net_image_dir, filename))
    # Resize it to desired format(227, 227, 3)
    img = cv2.resize(img, dsize=(227, 227), interpolation=cv2.INTER_CUBIC)
    return img
