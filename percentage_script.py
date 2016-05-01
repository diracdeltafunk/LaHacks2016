import tarfile
import re

tar = tarfile.open("pro.tar.gz", 'r:gz')
x, tot = 0.0, 0.0
with open('filenames.txt', 'r') as f:
    for filename in f.read().split('\n'):
        member = tar.getmember("pro/" + filename)
        f = tar.extractfile(member)
        data = str(f.read())
        if '+' in data:
            x += 1
        tot += 1

print(x / tot)
