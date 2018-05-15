import os
import tarfile

from pathlib import Path
import requests
from tqdm import tqdm

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


print('Size of wid -> urls file : {:8.2f} Mb'.format(os.path.getsize(image_net_urls_file)/1024/1024))
print('Size of wid -> word file : {:8.2f} Mb'.format(os.path.getsize(image_net_word_file)/1024/1024))


##########################################################################
# 2. Download two files given by ImageNet
##########################################################################


def download_wid2urls_file():
    print('Downloading Image Net WID -> Image Urls Mappings file...')
    r = requests.get(wids_url, stream=True)
    with open(image_net_urls_file, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        print('Total content length :', total_length)
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
        print('Total content length :', total_length)
        for chunk in tqdm(r.iter_content(chunk_size=1024), total=(total_length / 1024) + 1, unit='KB'):
            f.write(chunk)
            f.flush()


# Download second file
if not os.path.exists(image_net_word_file) or os.path.getsize(image_net_word_file) != 2655750:
    download_wid2types_file()
    assert os.path.getsize(image_net_word_file) == 2655750


##########################################################################
# 3. Extract urls from download zipped file
##########################################################################


def get_url2name():
    # Extract datasets
    print('Extracting urls from imagenet_fall11_urls.tgz...')
    url2name = {}
    with tarfile.open(image_net_urls_file) as tar:
        f = tar.extractfile('fall11_urls.txt')
        while True:
            try:
                filename, url = f.readline().decode().strip().split('\t')
                url2name[url] = filename
            except ValueError:
                break
    return url2name


url2name = get_url2name()
print('ImageNet contains {} images in total'.format(len(url2name)))


##########################################################################
# 4. Download Images
##########################################################################


def download_image(url, filename):
    try:
        print('Downloading', filename, 'from', url, '...')
        with open(os.path.join(image_net_image_dir, filename+'.jpg'), 'wb') as f:
            f.write(requests.get(url).content)
    except ConnectionError:
        print('Download Failed', url)
