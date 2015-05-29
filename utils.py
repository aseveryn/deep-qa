import scipy.io
import numpy
import re
import os
import pandas as pd
import subprocess
from datetime import datetime


def load_SST_data(path, include_sentence_size=False):
  print "Loading data from", path
  data = scipy.io.loadmat(path)
  vocab_emb = data['vocab_emb'].T

  # Pad the first row, s.t. we can index with original values from Matlab
  # (which start from 1) and the last row (empty).
  vocab_size = vocab_emb.shape[1]
  print 'vocab_size', vocab_size
  vocab_emb = numpy.vstack([numpy.zeros(vocab_size), vocab_emb, numpy.zeros(vocab_size)])
  print "Vocabulary", vocab_emb.shape, vocab_emb.dtype

  train_lbl = data['train_lbl'][:,0].astype('int32')-1
  train = data['train'].astype('int32')

  valid = data['valid'].astype('int32')
  valid_lbl = data['valid_lbl'][:,0].astype('int32')-1

  test = data['test'].astype('int32')
  test_lbl = data['test_lbl'][:,0].astype('int32')-1

  # print train[:3]

  if include_sentence_size:
    train_sentence_size = data['train_lbl'][:,1][:,numpy.newaxis].astype('int32')
    train = numpy.hstack([train, train_sentence_size])

    valid_sentence_size = data['valid_lbl'][:,1][:,numpy.newaxis].astype('int32')
    valid = numpy.hstack([valid, valid_sentence_size])

    test_sentence_size = data['test_lbl'][:,1][:,numpy.newaxis].astype('int32')
    test = numpy.hstack([test, test_sentence_size])

  return [(train, train_lbl), (valid, valid_lbl), (test, test_lbl)], vocab_emb



def is_ascii(s):
  return all(ord(c) < 128 for c in s)


NUM_ITEMS = 5

def iter_tweets(fname):
  print "Processing", fname
  for i, line in enumerate(open(fname), 1):
    items = line.strip().split('\t')
    if len(items) != NUM_ITEMS:
      print "Skipping line: {}, wrong number of fields: {}; expected: {}".format(i, len(items), NUM_ITEMS)
      continue
    label, docid, tokens, pos_tags, text = items
    # label = int(label)
    if label == 'unknwn':
      label = 0
    tokens = tokens.lower().split()
    pos_tags = pos_tags.split()
    clean_tokens = []
    for t, pos in zip(tokens, pos_tags):
      t = re.sub(r'(.)\1+', r'\1\1', t)  # Replace words with repeating characters
      t = re.sub(r'\d', '0', t)
      if not is_ascii(t):
        continue
      elif pos == '@':
        t = '@@'
      elif pos == 'U':
        t = '<URL>'
      elif pos == '$':
        t = '0'
      elif pos == 'G':
        t = 'G'
      clean_tokens.append(t)
    yield label, clean_tokens


def iter_tweets_term(fname):
  print "Processing", fname
  for i, line in enumerate(open(fname), 1):
    items = line.strip().split('\t')
    tokens, pos_tags, docid1, docid2, begin, end, label, text, _, _ = items
    if label == 'unknwn':
      label = 0
    tokens = tokens.lower().split()
    pos_tags = pos_tags.split()
    clean_tokens = []
    for t, pos in zip(tokens, pos_tags):
      t = re.sub(r'(.)\1+', r'\1\1', t)  # Replace words with repeating characters
      t = re.sub(r'\d', '0', t)
      if not is_ascii(t):
        continue
      elif pos == '@':
        t = '@@'
      elif pos == 'U':
        t = '<URL>'
      elif pos == '$':
        t = '0'
      elif pos == 'G':
        t = 'G'
      clean_tokens.append(t)
    yield label, clean_tokens, int(begin), int(end)


def iter_tweets_term_(fname):
  label2int = {'negative': 0, 'neutral': 1, 'positive': 2, 'unknwn': 0}

  print "Processing", fname
  for i, line in enumerate(open(fname), 1):
    items = line.strip().split('\t')
    docid1, docid2, begin, end, label, text = items

    label = label2int[label]
    tokens = text.decode('unicode-escape').encode('utf-8').lower().split()

    clean_tokens = []
    for t in tokens:
      t = re.sub(r'(.)\1+', r'\1\1', t)  # Replace words with repeating characters
      t = re.sub(r'\d', '0', t)
      if not is_ascii(t):
        continue
      elif t.startswith('@') and len(t) > 1:
        t = '@@'
      elif t.startswith('http://'):
        t = '<URL>'
      clean_tokens.append(t)
    yield label, clean_tokens, begin, end


def test_iter_tweets():
  fname = 'emoticon_tweets/all.tagged'
  for label, tokens in iter_tweets(fname):
    print label, tokens



def load_bin_vec(fname, words):
  """
  Loads 300x1 word vecs from Google (Mikolov) word2vec
  """
  print fname
  vocab = set(words)
  word_vecs = {}
  with open(fname, "rb") as f:
    header = f.readline()
    vocab_size, layer1_size = map(int, header.split())
    binary_len = numpy.dtype('float32').itemsize * layer1_size
    print 'vocab_size, layer1_size', vocab_size, layer1_size
    count = 0
    for i, line in enumerate(xrange(vocab_size)):
      if i % 100000 == 0:
        print '.',
      word = []
      while True:
        ch = f.read(1)
        if ch == ' ':
            word = ''.join(word)
            break
        if ch != '\n':
            word.append(ch)
      if word in vocab:
        count += 1
        word_vecs[word] = numpy.fromstring(f.read(binary_len), dtype='float32')
      else:
          f.read(binary_len)
    print "done"
    print "Words found in wor2vec embeddings", count
    return word_vecs


TASK_A = '/mnt/sdd/home/sovarm/nrc-twitter/data/semeval-2015/SemEval2015-task10-test-A-input.txt'
TASK_A_PROGRESS = '/mnt/sdd/home/sovarm/nrc-twitter/data/semeval-2015/SemEval2015-task10-test-A-input-progress.txt'

TASK_B = '/mnt/sdd/home/sovarm/nrc-twitter/data/semeval-2015/SemEval2015-task10-test-B-input.txt'
TASK_B_PROGRESS = '/mnt/sdd/home/sovarm/nrc-twitter/data/semeval-2015/SemEval2015-task10-test-B-input-progress.txt'


def score_semeval2015(pred_fn, nnet_outdir, data_dir):
  def predict(dataset):
    x_test = numpy.load(os.path.join(data_dir, 'semeval_{}_x.npy').format(dataset))
    print dataset, x_test.shape

    predictions = pred_fn(x_test)
    y2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    labels = [y2label[y] for y in predictions]
    return labels

  def write_predictions(input_fname, labels):
    data = pd.read_csv(input_fname, sep='\t', names=['id1', 'id2', 'label', 'text'])
    basename = os.path.basename(input_fname)
    data['label'] = labels
    data['id1'] = 'NA'
    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    outfile = os.path.join(nnet_outdir, '{}.output'.format(basename))
    print 'Writing to', outfile
    data.to_csv(outfile, sep='\t', header=False, index=False)
    return outfile

  labels = predict('test-2015')
  outfile = write_predictions(TASK_B, labels)

  labels = predict('test-2014')
  outfile = write_predictions(TASK_B_PROGRESS, labels)
  subprocess.call('/usr/bin/perl score-semeval2014-task9-subtaskB.pl "{}"'.format(outfile), shell=True)
  subprocess.call('/bin/cat "{}.scored"'.format(outfile), shell=True)



def score_semeval2015_term(pred_fn, nnet_outdir, data_dir):
  def predict(dataset):
    x_test = numpy.load(os.path.join(data_dir, 'semeval_{}_x.npy').format(dataset))
    x_test_term = numpy.load(os.path.join(data_dir, 'semeval_{}_x_term.npy').format(dataset))
    print dataset, x_test.shape

    predictions = pred_fn(x_test, x_test_term)
    y2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    labels = [y2label[y] for y in predictions]
    return labels

  def write_predictions(input_fname, labels):
    data = pd.read_csv(input_fname, sep='\t', names=['id1', 'id2', 'begin', 'end', 'label', 'text'])
    basename = os.path.basename(input_fname)
    data['label'] = labels
    data['id1'] = 'NA'
    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    outfile = os.path.join(nnet_outdir, '{}.output'.format(basename))
    print 'Writing to', outfile
    data.to_csv(outfile, sep='\t', header=False, index=False)
    return outfile

  labels = predict('test-2015')
  outfile = write_predictions(TASK_A, labels)

  labels = predict('test-2015-progress')
  outfile = write_predictions(TASK_A_PROGRESS, labels)
  subprocess.call('/usr/bin/perl score-semeval2014-task9-subtaskA.pl "{}"'.format(outfile), shell=True)
  subprocess.call('/bin/cat "{}.scored"'.format(outfile), shell=True)


def test_iter_tweets_term():
  fname = '/mnt/sdd/home/sovarm/nrc-twitter/data/semeval-2013-full/dev/input/twitter-dev-input-A.tsv'
  for items in iter_tweets_term(fname):
    print items


if __name__ == '__main__':
  # main()
  # test_iter_tweets()
  test_iter_tweets_term()