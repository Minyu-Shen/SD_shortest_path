from hashlib import new
from re import L
from this import d
import numpy as np
from collections import defaultdict, namedtuple
from network import Network
import scipy.stats as stats
from scipy import signal
import matplotlib.pyplot as plt


class Admissible_Path(object):
    def __init__(self):
        super(Admissible_Path, self).__init__()


class Algorithm(object):
    def __init__(self, network, search_type):
        super(Algorithm, self).__init__()
        self.search_type = search_type
        self.network = network
        self.inf = 1e+8
        # convoultion parameters: start, end, steps
        self.start = -50
        self.end = 50
        self.dx = 0.01
        self.grid = np.arange(self.start, self.end, self.dx)
        self.Admissible_Path = namedtuple(
            'Admissible_Path', ["pre_node", "pmf"])

    def plot(self, F, conved_F):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.grid, F, 'g-', lw=3, alpha=0.6, label='F')
        ax.plot(self.grid, conved_F, 'b-', lw=3,
                alpha=0.6, label='conved_F')
        plt.legend()
        plt.show()

    def determine_next_node(self):
        # Dijkstra
        if self.search_type == "BSFS":
            comp = 1e+8
            best_n = None
            for n in self.seq_list:
                if self.n_labels[n] <= comp:
                    comp = self.n_labels[n]
                    best_n = n
            return best_n
        # NOTE here I used a list to manipulate like "stack" and "heap"
        # the computational efficieny is not good. you can improve it easily, e.g., a deque structure
        # depth-first search
        elif self.search_type == "DFS":
            return self.seq_list[-1]
        # breadth-first search
        elif self.search_type == "BFS":
            return self.seq_list[0]

    def conv_link(self, f_pmf, link_info):
        if len(f_pmf) == 1:
            loc = int((f_pmf[0]-self.start) / self.dx)
            f_pmf = signal.unit_impulse(self.grid.shape, loc)

        if link_info.name == "deterministic":
            loc = int((link_info.mean-self.start) / self.dx)
            link_pmf = signal.unit_impulse(self.grid.shape, loc)
        elif link_info.name == "gamma":
            shape = 1 / (link_info.cv ** 2)
            scale = link_info.mean / shape
            gamma = stats.gamma(a=shape, scale=scale)
            link_pmf = gamma.pdf(self.grid) * self.dx
        conv_pmf = signal.fftconvolve(f_pmf, link_pmf, 'same')
        conv_pmf = conv_pmf/sum(conv_pmf)

        return conv_pmf

    def comp_dominance(self, conved_pmf, j):
        conved_F = np.cumsum(conved_pmf)
        dominating_ps = []
        drop_flag = False
        if len(self.n_label_path_dict[j]) == 0:
            return [], False
        for p_idx, path in self.n_label_path_dict[j].items():
            F = np.cumsum(path.pmf)
            if np.all(F - conved_F > -0.001):
                print("conved is dominated")
                drop_flag = True
            if np.all(conved_F - F > -0.001):
                print("conved is dominating")
                dominating_ps.append(p_idx)
            self.plot(F, conved_F)
        print("res:", dominating_ps, drop_flag)
        return dominating_ps, drop_flag

    def init_algo(self):
        ''' initialization '''
        # source and destination are set to be 0

        # maintain "node -> admissible paths"
        self.n_label_path_dict = defaultdict(dict)

        # pmf of the dummy path
        loc = int((0-self.start) / self.dx)
        pmf_ss = signal.unit_impulse(self.grid.shape, loc)
        ad_path = self.Admissible_Path(None, pmf_ss)

        self.n_label_path_dict[self.network.s] = {0: ad_path}
        self.seq_list = [self.network.s]  # starting from source node

    def main_loop(self):
        count = 0
        self.visited_nodes = []
        while len(self.seq_list) > 0:
            print("-------------------", count, "-------------------")
            # select the best node by current label
            i = self.determine_next_node()
            child_nodes = self.network.get_children(i)
            print("scan list:", self.seq_list)
            print("node i :", i)
            print("child js:", child_nodes)

            for j in child_nodes:
                # if j in self.visited_nodes:
                #     continue
                print("updating", j, "....")
                link_info = self.network.link_info_dict[(i, j)]
                for _, existing_path in self.n_label_path_dict[i].items():
                    conved_pmf = self.conv_link(existing_path.pmf, link_info)
                    dominating_ps, drop_flag = self.comp_dominance(
                        conved_pmf, j)
                    # if k is not dominated by any exisiting path
                    # add it
                    if not drop_flag:
                        # delete the path dominated by k
                        for p in dominating_ps:
                            self.n_label_path_dict[j].pop(p, None)

                        # add k to "Gamma" set
                        new_p = self.Admissible_Path(
                            pre_node=i, pmf=conved_pmf)
                        if len(self.n_label_path_dict[j]) == 0:
                            self.n_label_path_dict[j][0] = new_p
                        else:
                            exist_max = max(self.n_label_path_dict[j])
                            self.n_label_path_dict[j][exist_max+1] = new_p
                        # update scan node list
                        # if j not in self.seq_list and j is not self.network.d:
                        if j not in self.seq_list:
                            self.seq_list.append(j)

            self.seq_list.remove(i)
            self.visited_nodes.append(i)

            count += 1
            # if count >= 10:
            #     break

    def retrieve_res(self):
        for node, path_dict in self.n_label_path_dict.items():
            print('----------', node, "-----------")
            for _, path in path_dict.items():
                print(path.pre_node)

        # node = self.network.d
        # while node is not None:
        #     print(node)
        #     for _, path in self.n_label_path_dict[node].items():
        #         if path.pre_node != self.network.s:
        #             node = path.pre_node
        #         else:
        #             node = None

        # for node, paths in self.n_label_path_dict.items():
        #     print(node, len(paths))


if __name__ == "__main__":
    # network = Network("pentagram", source=1, destination=4)
    network = Network("Braess", source=0, destination=3)
    # "best-first search" is not defined for random travel time
    # search_type = "BSFS"
    search_type = "DFS"
    # search_type = "BFS"
    algo = Algorithm(network, search_type)
    algo.init_algo()
    algo.main_loop()
    algo.retrieve_res()
