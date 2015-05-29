
class Alphabet(dict):
  def __init__(self, start_feature_id=1):
    self.fid = start_feature_id

  def add(self, item):
    idx = self.get(item, None)
    if idx is None:
      idx = self.fid
      self[item] = idx
      self.fid += 1
    return idx

  def dump(self, fname):
    with open(fname, "w") as out:
      for k in sorted(self.keys()):
        out.write("{}\t{}\n".format(k, self[k]))


def test():
  import pickle

  a = Alphabet()
  print a.fid
  a.add('2')
  a.add('1')
  a.add('1')
  print a.fid, a
  pickle.dump(a, open('/tmp/tmp.pickle', 'w'))
  del a

  a = pickle.load(open('/tmp/tmp.pickle'))
  print a.fid, a
  a.add('4')
  print a.fid, a

  a = Alphabet(start_feature_id=0)
  a.add('4')
  print a

if __name__ == '__main__':
  test()