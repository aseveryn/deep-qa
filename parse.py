import re
import os
import numpy as np
import cPickle
import subprocess
from collections import defaultdict

from alphabet import Alphabet


UNKNOWN_WORD_IDX = 0


def load_data(fname):
  lines = open(fname).readlines()
  qids, questions, answers, labels = [], [], [], []
  num_skipped = 0
  prev = ''
  qid2num_answers = {}
  for i, line in enumerate(lines):
    line = line.strip()

    qid_match = re.match('<QApairs id=\'(.*)\'>', line)

    if qid_match:
      qid = qid_match.group(1)
      qid2num_answers[qid] = 0

    if prev and prev.startswith('<question>'):
      question = line.lower().split('\t')

    label = re.match('^<(positive|negative)>', prev)
    if label:
      label = label.group(1)
      label = 1 if label == 'positive' else 0
      answer = line.lower().split('\t')
      if len(answer) > 60:
        num_skipped += 1
        continue
      labels.append(label)
      answers.append(answer)
      questions.append(question)
      qids.append(qid)
      qid2num_answers[qid] += 1
    prev = line
  # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
  print 'num_skipped', num_skipped
  return qids, questions, answers, labels


def compute_overlap_features(questions, answers, word2df=None, stoplist=None):
  word2df = word2df if word2df else {}
  stoplist = stoplist if stoplist else set()
  feats_overlap = []
  for question, answer in zip(questions, answers):
    # q_set = set(question)
    # a_set = set(answer)
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)
    # overlap = float(len(word_overlap)) / (len(q_set) * len(a_set) + 1e-8)
    overlap = float(len(word_overlap)) / (len(q_set) + len(a_set))

    # q_set = set([q for q in question if q not in stoplist])
    # a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)
    df_overlap = 0.0
    for w in word_overlap:
      df_overlap += word2df[w]
    df_overlap /= (len(q_set) + len(a_set))

    feats_overlap.append(np.array([
                         overlap,
                         df_overlap,
                         ]))
  return np.array(feats_overlap)


def compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length):
  stoplist = stoplist if stoplist else []
  feats_overlap = []
  q_indices, a_indices = [], []
  for question, answer in zip(questions, answers):
    q_set = set([q for q in question if q not in stoplist])
    a_set = set([a for a in answer if a not in stoplist])
    word_overlap = q_set.intersection(a_set)

    q_idx = np.ones(q_max_sent_length) * 2
    for i, q in enumerate(question):
      value = 0
      if q in word_overlap:
        value = 1
      q_idx[i] = value
    q_indices.append(q_idx)

    #### ERROR
    # a_idx = np.ones(a_max_sent_length) * 2
    # for i, q in enumerate(question):
    #   value = 0
    #   if q in word_overlap:

    a_idx = np.ones(a_max_sent_length) * 2
    for i, a in enumerate(answer):
      value = 0
      if a in word_overlap:
        value = 1
      a_idx[i] = value
    a_indices.append(a_idx)

  q_indices = np.vstack(q_indices).astype('int32')
  a_indices = np.vstack(a_indices).astype('int32')

  return q_indices, a_indices


def compute_dfs(docs):
  word2df = defaultdict(float)
  for doc in docs:
    for w in set(doc):
      word2df[w] += 1.0
  num_docs = len(docs)
  for w, value in word2df.iteritems():
    word2df[w] /= np.math.log(num_docs / value)
  return word2df


def add_to_vocab(data, alphabet):
  for sentence in data:
    for token in sentence:
      alphabet.add(token)


def convert2indices(data, alphabet, dummy_word_idx, max_sent_length=40):
  data_idx = []
  for sentence in data:
    ex = np.ones(max_sent_length) * dummy_word_idx
    for i, token in enumerate(sentence):
      idx = alphabet.get(token, UNKNOWN_WORD_IDX)
      ex[i] = idx
    data_idx.append(ex)
  data_idx = np.array(data_idx).astype('int32')
  return data_idx



if __name__ == '__main__':
  # stoplist = set([line.strip() for line in open('en.txt')])
  # import string
  # punct = set(string.punctuation)
  # stoplist.update(punct)
  stoplist = None


  train = 'jacana-qa-naacl2013-data-results/train.xml'
  train_all = 'jacana-qa-naacl2013-data-results/train-all.xml'
  train_files = [train, train_all]

  for train in train_files:
    print train

    dev = 'jacana-qa-naacl2013-data-results/dev.xml'
    test = 'jacana-qa-naacl2013-data-results/test.xml'

    train_basename = os.path.basename(train)
    name, ext = os.path.splitext(train_basename)
    outdir = '{}'.format(name.upper())
    print 'outdir', outdir

    if not os.path.exists(outdir):
      os.makedirs(outdir)

    # all_fname = train
    all_fname = "/tmp/trec-merged.txt"
    files = ' '.join([train, dev, test])
    subprocess.call("/bin/cat {} > {}".format(files, all_fname), shell=True)

    # qids, questions, answers, labels = load_data(all_fname, stoplist)
    qids, questions, answers, labels = load_data(all_fname)

    ### Compute document frequencies.
    seen = set()
    unique_questions = []
    for q, qid in zip(questions, qids):
      if qid not in seen:
        seen.add(qid)
        unique_questions.append(q)

    docs = answers + unique_questions
    word2dfs = compute_dfs(docs)
    print word2dfs.items()[:10]
    #########

    alphabet = Alphabet(start_feature_id=0)
    alphabet.add('UNKNOWN_WORD_IDX')

    add_to_vocab(answers, alphabet)
    add_to_vocab(questions, alphabet)

    basename = os.path.basename(train)
    cPickle.dump(alphabet, open(os.path.join(outdir, 'vocab.pickle'), 'w'))
    print "alphabet", len(alphabet)

    dummy_word_idx = alphabet.fid

    q_max_sent_length = max(map(lambda x: len(x), questions))
    a_max_sent_length = max(map(lambda x: len(x), answers))
    print 'q_max_sent_length', q_max_sent_length
    print 'a_max_sent_length', a_max_sent_length

    # Convert dev and test sets
    for fname in [train, dev, test]:
      print fname
      # qids, questions, answers, labels = load_data(fname, stoplist)
      qids, questions, answers, labels = load_data(fname)

      overlap_feats = compute_overlap_features(questions, answers, stoplist=None, word2df=word2dfs)
      overlap_feats_stoplist = compute_overlap_features(questions, answers, stoplist=stoplist, word2df=word2dfs)
      overlap_feats = np.hstack([overlap_feats, overlap_feats_stoplist])
      # overlap_feats = compute_overlap_features(questions, answers, None)
      # print overlap_feats[:10]
      print 'overlap_feats', overlap_feats.shape

      qids = np.array(qids)
      labels = np.array(labels).astype('int32')

      _, counts = np.unique(labels, return_counts=True)
      print counts / float(np.sum(counts))
      print "questions", len(np.unique(qids))
      # print "questions", len(qids)
      print "pairs", len(labels)

      # stoplist = None
      q_overlap_indices, a_overlap_indices = compute_overlap_idx(questions, answers, stoplist, q_max_sent_length, a_max_sent_length)
      # print q_overlap_indices[:3]
      # print a_overlap_indices[:3]

      questions_idx = convert2indices(questions, alphabet, dummy_word_idx, q_max_sent_length)
      answers_idx = convert2indices(answers, alphabet, dummy_word_idx, a_max_sent_length)
      print 'answers_idx', answers_idx.shape

      basename, _ = os.path.splitext(os.path.basename(fname))
      np.save(os.path.join(outdir, '{}.qids.npy'.format(basename)), qids)
      np.save(os.path.join(outdir, '{}.questions.npy'.format(basename)), questions_idx)
      np.save(os.path.join(outdir, '{}.answers.npy'.format(basename)), answers_idx)
      np.save(os.path.join(outdir, '{}.labels.npy'.format(basename)), labels)
      np.save(os.path.join(outdir, '{}.overlap_feats.npy'.format(basename)), overlap_feats)

      np.save(os.path.join(outdir, '{}.q_overlap_indices.npy'.format(basename)), q_overlap_indices)
      np.save(os.path.join(outdir, '{}.a_overlap_indices.npy'.format(basename)), a_overlap_indices)
