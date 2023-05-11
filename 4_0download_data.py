from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_covtype
from sklearn.datasets import load_svmlight_file
# from sklearn.datasets import fetch_mldata
import pickle
import urllib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# mnist = fetch_mldata("MNIST original")
# pickle.dump(mnist, open("mnist.pickle", "wb"))

target_page = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2'
with urllib.request.urlopen(target_page) as response:
    with open('ijcnn1.bz2', 'wb') as W:
        W.write(response.read())

target_page = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata'
cadata = load_svmlight_file(urllib.request.urlopen(target_page))
pickle.dump(cadata, open("cadata.pickle", "wb"))

covertype_dataset = fetch_covtype(random_state=101, shuffle=True)
pickle.dump(covertype_dataset, open(
    "covertype_dataset.pickle", "wb"))

newsgroups_dataset = fetch_20newsgroups(shuffle=True,
                                        remove=('headers', 'footers', 'quotes'), random_state=6)
pickle.dump(newsgroups_dataset, open("newsgroups_dataset.pickle", "wb"))
