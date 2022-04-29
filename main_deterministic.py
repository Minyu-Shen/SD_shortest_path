import numpy as np
from collections import defaultdict
from network import Network


class Algorithm(object):
    def __init__(self, network, search_type):
        super(Algorithm, self).__init__()
        self.search_type = search_type
        self.network = network
        self.inf = 1e+8

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

    def init_algo(self):
        ''' initialization '''
        # source and destination are set to be 0
        self.n_labels = [self.inf if n not in [self.network.s]
                         else 0 for n in range(self.network.n_num)]
        self.seq_list = [self.network.s]  # starting from source node
        self.predecessor = [None for n in range(self.network.n_num)]

    def main_loop(self):
        count = 0
        self.visited_nodes = []
        while len(self.seq_list) > 0:
            print("--------------------------",
                  count, "--------------------------")
            # select the best node by current label
            i = self.determine_next_node()
            child_nodes = self.network.get_children(i)
            print("seq list:", self.seq_list)
            print("node removed:", i)
            print("children node:", child_nodes)
            self.visited_nodes.append(i)

            self.seq_list.remove(i)
            for j in child_nodes:
                if j in self.visited_nodes:
                    continue
                link_info = self.network.link_info_dict[(i, j)]
                mean_time = link_info.mean
                # print('!!!', i, j,
                #       self.n_labels[i] + mean_time, self.n_labels[j])
                if self.n_labels[i] + mean_time < self.n_labels[j]:
                    print("update:", j)
                    self.n_labels[j] = self.n_labels[i] + mean_time
                    self.predecessor[j] = i
                    if j not in self.seq_list:
                        self.seq_list.append(j)

            print("labels:", self.n_labels)
            print("predecessor trace", self.predecessor)

            # count += 1
            # if count > 4:
            #     break


if __name__ == "__main__":
    source = 1  # source node
    destination = 4  # destination node
    network = Network("pentagram", source, destination)
    # "best-first search"
    search_type = "BSFS"
    search_type = "DFS"
    # search_type = "BFS"
    algo = Algorithm(network, search_type)
    algo.init_algo()
    algo.main_loop()
