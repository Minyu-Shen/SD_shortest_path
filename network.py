import numpy as np
from collections import namedtuple, defaultdict


class Network(object):
    def __init__(self, name, source, destination):
        super(Network, self).__init__()
        self.TT_Dist = namedtuple("TT_Distribution", ["mean", "cv", "name"])
        self.s = source
        self.d = destination
        if name == "pentagram":
            ''' a pentagram network (without randomness) for testing label correcting algorithm'''
            self.n_num = 5
            self.n2n_Delta = np.ones((5, 5))
            self.n2n_mean = np.array([[-1, 6, 5, 2, 2], [6, -1, 0.5, 5, 7], [5, 0.5, -1, 1, 5], [
                                     2, 5, 1, -1, 3], [2, 7, 5, 3, -1]], dtype=np.float32)
            self.n2n_cv = np.ones((5, 5)) / 3.0
            self.n2n_name = {(i, j): "deterministic" for i in range(5)
                             for j in range(5)}
            # self.n2n_name = {(i, j): "gamma" for i in range(5)
            #                  for j in range(5)}

            # assert symmetric
            assert np.all(np.abs(self.n2n_mean-self.n2n_mean.T)
                          < 1e-8), "input is wrong"
        elif name == "Braess":
            self.n_num = 4
            self.n2n_Delta = np.array(
                [[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
            self.n2n_mean = np.array([[-1, 1, 2, -1], [-1, -1, 0.5, 3], [-1, -1, -1, 1], [-1, -1, -1, -1]])
            self.n2n_cv = np.ones((4, 4)) / 3.0
            self.n2n_name = {(i, j): "deterministic" for i in range(4)
                             for j in range(4)}

        self.link_info_dict = self.build_link_info()

    def get_children(self, parent_n):
        children = [n for n, v in enumerate(
            self.n2n_Delta[parent_n]) if v == 1 and n != parent_n]
        return children

    def build_link_info(self):
        link_info_dict = defaultdict(self.TT_Dist)
        node_num = self.n2n_Delta.shape[0]
        for i in range(node_num):
            for j in range(node_num):
                if self.n2n_Delta[i, j] == 1:
                    mean = self.n2n_mean[i][j]
                    cv = self.n2n_cv[i][j]
                    name = self.n2n_name[(i, j)]
                    link_info_dict[(i, j)] = self.TT_Dist(mean, cv, name)
        return link_info_dict
