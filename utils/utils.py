
class idsMap:
    def __init__(self, ids=[], name=''):
        print('__init__===>{}'.format(name))
        self.idx2idmap = {}
        self.id2idxmap = {}
        self.name = name
        self.update(ids)
        
    def len(self):
        return len(self.id2idxmap)

    def update(self, ids):
        print('-*-'*20)
        print('{}::::before update map size={}'.format(self.name, self.len()))
        for term in ids:
            if term in self.idx2idmap: pass
            else:
                cur_size = self.len()
                self.idx2idmap[term] = cur_size
                self.id2idxmap[cur_size] = term
        print('{}::::after updatemap size={}'.format(self.name, self.len()))
        print('-*-'*20)

    def idx2id(self, arr):
        return [self.idx2idmap[t]for t in arr]
         
    def id2idx(self, arr):
        return [self.id2idxmap[t]for t in arr]
