# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + id="BjrDxi2V7RWT"
# !pip install transformers
# !pip install simpletransformers
# !pip install scikit-learn
# !pip install wandb

# !echo "--- Disk ---"
# !df -h
# !echo ""
# !echo "--- CPU ---"
# !cat /proc/cpuinfo
# !echo ""
# !echo "--- Memory ---"
# !cat /proc/meminfo
# !echo ""
# !echo "--- GPU ---"
# !nvidia-smi -L

# Mount Google Drive with input data
from google.colab import drive
drive.mount('/content/drive')

# + colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["f749e392af6c464689bd421142677e27", "87a9d819d10042e8acf1d3e56c4ff30b", "a6c68f7daebf4ff29a5038fe16fd6276", "bd554f1cd7af4ee19c6215338603996d", "d4f818dac8ad444fa8a8a79d354da5d3", "52ce17b597054a3986558a4a5533dc68", "ca80b748a7ec421b8070e17f82b684ce", "272c57a300ce43b9b1721a1cc87979f1", "6e2b66a5a6c64805951f3fb4f196ca20", "b36c7b327a1544cea862dfce08f39050", "d5cd447853684e2bac444d2ca8bc5f4e", "c83278fd8f11465cab1e9f17a5849cfa", "6cd3c2321f7e4bbbb3146271d130337f", "96488cdedd6b40e5998b179b4963663a", "b5cff864f2d441d29afc80ad6025dfc3", "e577e85eccbf4d94a8f9179504cd4975", "5d249ec560674c5ea60b39102d339948", "88324d9559ce484c9a82b7b0375fa34e", "2b9a35a293bd49c29827cf10dce43b89", "7d1c3b99fdba43e5a7163985ef456475", "8ce1f7a6ed4b4117b3a25c2e69af5106", "d8c40274569843b8adac153e8d5eda1e", "427db8365c45403cba1297a68d0aa67c", "4c8ee1020bd2414fb6ec6f1cceaba880", "d25e31d90f6b450ca4613f2b65fa2707", "983efd39447b43779ef5ed72514f6655", "58c7ab606b524c48bd6e8d580eb0be4c", "ef043aae1c504f8a914dcc21c7412b4c", "a500984c5baa4d6bb386b44994bbbc8a", "2f69604168e74b7aa8927541cd554217", "06c293ebdf484a858a98034390423178", "b5bb9e3b384c4640962a056e10c134bd", "9bbf74947d7749e181ae00a3ca6785c7", "5b11b253a7dd4c19948489a0a49bc75c", "5f93ce8813db41a8a54089dc5fe80f65", "3ebe34e5f0834bc2893fd78ae02b1525", "8959719c321c42d8b6346cbfe4a5c408", "691730c2f7994409bf15009a0d933f5b", "72604408691b47569b357c7106e31c13", "14b204aa73394321a8be7832d7695dff", "30a6626a24754f2ea3a52226eebbbb34", "219796b1888548d8b7fbc8919a54e678", "de3c81587efc49deba272185d92d01cb", "1ad7f9b31e8847de8edc2946fbc90014", "943ae992a9ca4c2baab4cd79609974bd", "869b115a96c74a2c9b04f76f0b58d3e6", "b706f1ef964b4c58a11416ff9028020e", "99820b6f73e34e049a555829c0c19b9b", "25b8909a62c641e69cbb3e59d5c6f2b4", "f20ef926f3a34169a171890c27f47235", "b368c17213e54d3b9b82bcf64fb2f59c", "8308171cd38d4e2da08d4e65a9bcca21", "c8cd54e539154e9eab590396cf5f9d0e", "d67c2207681f4449a3cdaf6f75e778c8", "32f64b0495b3421eb74fc867f78c77e5", "56c342de03764f7cb98657a056e8ef10", "bef1e8ed613742b2acc7c0efdea0504a", "bf10a35c70854f299253929dac8210bc", "71d97bd4b8694ccd9d697fa063a120f4", "0e1df5596cb1452eb81a8bf684f1be68", "5561d3e0f1f54ab2b94b6c9212398ad7", "81ca48fbcf4d443f92c334c89901b460", "eefda32b8be14ae6a915fceae480eece", "490b8c19ec954079958bc25e3339b1f3", "70a8d1e7df2c48678666f8871905e0f6", "d9836b84ac404127af41ff1de2358900", "bc094e77836148cd807993bda00879ab", "837d19b2f7d5409cb843311d1ef20b12", "f2e812a24ae94e9e81749ca591449137", "c7eab51e0fae40678603513a0ce6c933", "f2f6e647bb164f94a32910a2ae2d0e3f", "fa884f3c437f45238f6ca956baaa7e7c", "1897ec92c99a4e3abbf5e114b9134fc0", "0f9d42c52b4b4205af378923ac31ce19", "51fdd57c7b3f4a7eb3070931bb3be530", "39e58733fcde434fba064b6bc159c482", "9c235e904201477cbb8794b10d421989", "b29790771c564f00924d2da43e2fcf08", "0fb6f336365c40058d165906aafdac33", "47729673da4e4ba3b1802245d9bca975", "5e937ba884ba46f09007a9906551b578", "2953886eeeb14396874274edf5798506", "5f0ac63a7637406c90eb08bd3e1ecdf3", "16501a1f1a284b0dbb6c3b4aca6abf72", "012c3bac443344afbc231e6e7d84280d", "ccade934245246ad92c25388920d6443", "182898ec39354e50a9023e79c34506bd", "6310d0c0128744feb78d1fe83a2cf129", "1abf78f1a5844a6295318ccf27ba8f9c", "97ec30c547b443e3992cbfbb706d39d1", "038699c6700d4b4caab1bdb443169e99", "30c3187feec942a68ce22a6a45961fc0", "e8f22fb937974e39bda6935884b2ba53", "350a0ae5800749efb050b9f595439af6", "9ac637910ce04c6dbb25857613aa0884", "6755ca6470bf418c817f8bb934b9986d", "65c40a858154466090bc1b2f5ba5665f", "e88f91e11f1b42c39a1e8810015759cd", "076ed344202a40b7b1c3cf6ff7b1b8ca", "6c8b38b3ec404dc69b6b12cedeee134c", "ab45baadbd07412cadde5d19ac2d75be", "5c9b77fdb63743838017d162d2febef1", "7a5ac148860743e9b5a46559168a0a46", "f5f7c34f2df64a14bbb86537f2ce8438", "9f630afb754e4ce194e6031a9f296518", "034d472ed49c45f1ae85c1db7c8dab5b", "bf89ffb257a145f0b97a0ddf18fb1fc7", "7b4e6626a46b40ab867d958b553d5ab3", "905f7876434d41a1bd416c1bd39782b9", "f27ef08b20a6409982e09c79ea30f0c7", "581700d1d1fa4362b6d729daa9f168b9", "d4371afc5b7c4d2495cdfe1874a2001b", "e04c1233776549768afb59fe788c3f83", "98cb7de289a040f8bcbdb2299c3556b1", "11ee508c04a1449d921e8e08bfb4472d", "1c20fbfdd51643e884e59379495d5c9c", "94a473e5ac8d43e590b29d7b0d95a592", "854898f00521487cbd5b64af3d88fc2d", "37ed42e3af3f49269632b772eea00174", "3eb05bcc5acd4a0781ff34701c086396", "220149b3782945edacb7508b10abf67f", "9139e08650e44fb58a20c5a8491c6740", "29c5b885b6214b1d952ce607f3642b2f", "5aea4087d15441fdb8907467cd13b93c", "b9caf99c355048a0abc1f8f1fdb2c4d3", "88f3460351034a23b792f61cb404f47a", "32dbc77591624fb8bd210addf9693521", "3b8b0a7331194654af6487a57bf45ad3", "f8fb1ab23ba84fda887a0f978ce05532", "631d8c60eaf54ca3befd57046cf37645", "d54336495f4242cabf3135e48cab6611", "c7a8f83d41af4d788889e974ac4b085a", "6ad6a57ea0e14d4c9cd30b5f5a240e8d", "24973fd622b64e52a327d11432eaadd6", "24e11af48391407591b8906f392f89af", "2d8e1352c9eb468bafe0e6c8788d29f5", "bb470073a6b84afeac7abd11f7cad6ba", "b2a7b8f70c2845ae8ad8e266ba471112", "e5b9f7ddff534c9cae8705c7a99425b2", "cd33a988a3ca4a138a06995906938a38", "22d1798717b847aa8574f6478369d358", "ec758f7e2957447db5a40151b4577a57", "f12d9ca923b54230bd0ce562e7538110", "e642b56e4208410b89e302061c9766ee", "b2c5e6101214463d9cc9bb7525876858", "32b2227a878d4f62a177ae70d6be3d83", "c0c3f93ea7ec47a9b25c07758caeae12", "d4987e7b3bf941b2925fa4c38ba557a6", "62a2eb2c6a694f6983fc7bf20c235303", "f6ac1ca2aac146d0a85b5a562b5aaee1", "d8595a1f9fc1460a81a30b6dccd1c505", "f8cc0249d10e48c88c8111f80e2a74d2", "4184a93761404200861006a64f58ac24", "4b866f0a20b94ee18789348cc2a7e512"]} id="6WvUJV0H76aZ" outputId="12730977-0012-4976-c7f7-88909667f43a"
from simpletransformers.classification import (
    ClassificationModel, ClassificationArgs
)
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics as metrics
import pandas as pd
import random as rand
import wandb

# initialize for deterministic results
seed = 0
rand.seed(seed)

# load data
path = '/content/drive/My Drive/Colab Notebooks/Liter/correlations/corresult4.csv'
data = pd.read_csv(path, sep = ',')
data = data.sample(frac=1, random_state=seed)
data.columns = ['dataid', 'datapath', 'nrrows', 'nrvals1', 'nrvals2', 
                'type1', 'type2', 'column1', 'column2', 'method',
                'coefficient', 'pvalue', 'time']

# divide data into subsets
pearson = data[data['method']=='pearson']
spearman = data[data['method']=='spearman']
theilsu = data[data['method']=='theilsu']

# generate and print data statistics
nr_ps = len(pearson.index)
nr_sm = len(spearman.index)
nr_tu = len(theilsu.index)
print(f'#Samples for Pearson: {nr_ps}')
print(f'#Samples for Spearman: {nr_sm}')
print(f'#Samples for Theil\'s u: {nr_tu}')

# |coefficient>0.5| -> label 1
def coefficient_label(row):
  if abs(row['coefficient']) > 0.5:
    return 1
  else:
    return 0
pearson['label'] = pearson.apply(coefficient_label, axis=1)
spearman['label'] = spearman.apply(coefficient_label, axis=1)
theilsu['label'] = theilsu.apply(coefficient_label, axis=1)

rc_p = len(pearson[pearson['label']==1].index)/nr_ps
rc_s = len(spearman[spearman['label']==1].index)/nr_sm
rc_u = len(theilsu[theilsu['label']==1].index)/nr_tu
print(f'Ratio correlated - Pearson: {rc_p}')
print(f'Ratio correlated - Spearman: {rc_s}')
print(f'Ratio correlated - Theil\s u: {rc_u}')

# split data into training and test set
def def_split(data):
  x_train, x_test, y_train, y_test = train_test_split(
      pearson[['column1', 'column2']], pearson['label'],
      test_size=0.2, random_state=seed)
  train = pd.concat([x_train, y_train], axis=1)
  test = pd.concat([x_test, y_test], axis=1)
  return train, test

def ds_split(data):
  counts = data['dataid'].value_counts()
  print(f'Counts: {counts}')
  print(f'Count.index: {counts.index}')
  print(f'Count.index.values: {counts.index.values}')
  print(f'counts.shape: {counts.shape}')
  print(f'counts.iloc[0]: {counts.iloc[0]}')
  nr_vals = len(counts)
  nr_test_ds = int(nr_vals * 0.2)
  print(f'Nr. test data sets: {nr_test_ds}')
  ds_ids = counts.index.values.tolist()
  print(type(ds_ids))
  print(ds_ids)
  test_ds = rand.sample(ds_ids, nr_test_ds)
  print(f'TestDS: {test_ds}')
  def is_test(row):
    if row['dataid'] in test_ds:
      return True
    else:
      return False
  data['istest'] = data.apply(is_test, axis=1)
  train = data[data['istest'] == False]
  test = data[data['istest'] == True]
  print(f'train.shape: {train.shape}')
  print(f'test.shape: {test.shape}')
  print(train)
  print(test)
  return train[['column1', 'column2', 'label']], test[['column1', 'column2', 'label']]

train, test = ds_split(pearson)
train.columns = ['text_a', 'text_b', 'labels']
test.columns = ['text_a', 'text_b', 'labels']
print(train.head())
print(test.head())

model_args = ClassificationArgs(num_train_epochs=10, train_batch_size=40,
                                overwrite_output_dir=True, manual_seed=seed,
                                evaluate_during_training=True, no_save=True,
                                wandb_project='CorrelationPredictionv1')
model = ClassificationModel("roberta", "roberta-base", weight=[1, 2],
                            use_cuda = True, args=model_args)
model.train_model(train_df=train, eval_df=test, acc=metrics.accuracy_score, 
    rec=metrics.recall_score, pre=metrics.precision_score, f1=metrics.f1_score)
wandb.join()
#output_dir='/content/drive/My Drive/Colab Notebooks/Liter/correlations/models'

# + colab={"base_uri": "https://localhost:8080/", "height": 431, "referenced_widgets": ["c14a7fcbc7154cfa8159e35f53e248fe", "96e98ccc20d84020a0cf13934c8f000c", "2416a74be5ad492b9c37f73fead5cee0", "dfbca7a918d2494e97cfceb2da3470e2", "9bcbaf346bc84db793b3a372cfbe0bb4", "09461fad07f54bc6bec93a3f6cf729a7", "09905a6f9ae4447a95106c2f95d68df4", "f2a3acf8c9074266b9bed0a042e25c3d", "ac04122fc46847c09a9afc79ee20a1c8", "4cb4b79eb020494d913f98f6f29fe11e", "db561081b7f94a0680300a0c97987b5b", "4cd5d9bd988e42edba56fffc566ec2e1", "4cb04e7c35bf46acab1dcaf01916067d", "30addc08b5dc47b7b810703894a0964f", "819d10cd8cbb43b5a678d8f410824d3c", "2f3abf392d384e1eb4fd1abd782badfd", "27afcde85f7c405f94b5ace454947e0d", "91028f793a6f4079a3fd69d7001f5e57", "6024c4c5c9e24fdeb2b76d3a3c4f17e6", "a9789f427d234b619434898aa0b3d4f0", "19af0fbc18bb4da69e5f9fd85dd8f3a0", "5bacca183de84e0ab58f5e7b653ab59a", "1a357757bf7845c687d066f57ec9312e", "a42471c4feae4034b0b1bad116847204", "a08c6442fb574727a067742616c4d1ac", "e159a15c5f164461a7c9357d1614a748", "155288aa57cb4c45867873f7714e0d12", "486df80d65d54e8eb3f44dd947522cd2", "4f41725a3610415dad84c2e66a12df64", "0485fa54b8354ee48769efbba7608faa", "c124c35f54844618928bce25383f9ab6", "9b3d0b92f0684182a53afac23d316d4a", "b819f7a5ff584073ae2cad63f294578e", "ab4631ba36254151a73fcfe5cdf905ca", "dbaadd36132647f7aa4e7069d2c99ca8", "7d311f17790a44aa9fa62a6fc8fcf11c", "fd91b7c815c44080a3891c04978d2b98", "1fceb4d5ed024bdf82fd2f58a496f854", "afcff9e67aff4199bc80f564292fc76a", "d9da31a0a45d4c5d8cf2dbbf1d77b903", "541db5d0e0aa4c4881847f0d3b6084e4", "a82cfcb6c60147d5920746c55bfe87b9", "72d7f88ffdf04b0fb3a8fc3640e974c5", "91e7a795e29d4e18b1257ffaedff80e6"]} id="BO2DgrywIivg" executionInfo={"status": "ok", "timestamp": 1609523808144, "user_tz": 300, "elapsed": 74109, "user": {"displayName": "Immanuel Trummer", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gg2qNKuNW8WA0c838ZIfZt2X16sK_rB7ph-Yxb2OQ=s64", "userId": "08659789901574647850"}} outputId="2b8461ed-be75-4ca7-93da-8be6240cd0dd"
import torch
from simpletransformers.classification import ClassificationModel
import sklearn.metrics as metrics

model = ClassificationModel('roberta', '/content/drive/My Drive/Colab Notebooks/Liter/correlations/models/checkpoint-865-epoch-1')
result, outputs, failures = model.eval_model(
    test, acc=metrics.accuracy_score, rec=metrics.recall_score, 
    pre=metrics.precision_score, f1=metrics.f1_score)
print(result)
test_samples = []
for idx, r in test.iterrows():
  test_samples.append([r['text_a'], r['text_b']])
pred = model.predict(test_samples)
test['pred'] = pred[0]
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
print(test)
