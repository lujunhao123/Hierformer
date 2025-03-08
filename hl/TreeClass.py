import numpy as np
import pandas as pd
class Node(object):
    """A Node class object to encapsulate node related attributes."""

    def __init__(self,
                 id: str,
                 parent: list or None,
                 children: list or None,
                 **kwargs):
        self.id = id
        self.parent = parent
        self.children = children
        self.k = kwargs['k'] if 'k' in kwargs else None
        self.type = kwargs['type'] if 'type' in kwargs else \
            'root' if parent is None else 'leaf' if children is None else 'node'

    def __lt__(self, other):
        return self.k > other.k

class TreeInitialization(object):
    """A Tree Initialization class object to encapsulate tree structure related attributes and methods."""

    def __init__(self,
                 structural_input_information: set or dict or tuple,
                 dimension: str):

        if dimension not in ['spatial', 'temporal', 'spatiotemporal']:
            raise ValueError("Input dimension specified is incorrect. Possible dimensions encapsulate: spatial, "
                             "temporal, spatiotemporal.")
        self.dimension = dimension

        # Information derived from nodes
        if type(structural_input_information) is set and self.dimension == 'spatial':
            self.nodes = structural_input_information
            self.root = [node for node in self.nodes if node.type == 'root'][0]
            self.leaves = [node for node in self.nodes if node.type == 'leaf']
            self.leaves_label = [node.id for node in self.nodes if node.type == 'leaf']
            self.n = len(self.nodes)
            self.m = len(self.leaves)
            self.identify_k_values_of_nodes_from_nodes()
            self.S, self.y_ID = self.build_S_from_treestr(self.nodes)
            #self.leaves_label = [node for i, node in enumerate(self.y_ID) if self.S[i, :].sum() == 1]
            self.k_level_map = self.build_k_level_map_from_S()
            self.K = np.max(list(self.k_level_map.keys()))
            self.k_level_y = np.sum(self.S, axis=1)

        # Information derived from k_level_resamplings
        elif type(structural_input_information) is dict and self.dimension == 'temporal':
            self.k_level_resamplings = structural_input_information
            self.k_level_map = self.create_k_level_map_from_klvlresamplg(self.k_level_resamplings)
            self.K = np.max(list(self.k_level_map.keys()))
            self.leaves_label = self.k_level_map[1]
            self.horizon = pd.Timedelta(self.leaves_label[0].split('_')[0]) * len(self.leaves_label)
            self.nodes = self.build_treestr_from_kmap(self.k_level_map)
            self.root = [node for node in self.nodes if node.type == 'root'][0]
            self.n = len(self.nodes)
            self.m = len(self.leaves_label)
            self.S, self.y_ID = self.build_S_from_treestr(self.nodes)
            self.leaves = [node for node in self.nodes if node.type == 'leaf']
            self.k_level_y = np.sum(self.S, axis=1)

        # Information derived from Summation matrix
        elif type(structural_input_information) is tuple:
            self.S = structural_input_information[0]
            self.y_ID = structural_input_information[1]
            self.n = np.shape(self.S)[0]
            self.m = np.shape(self.S)[1]
            self.leaves_label = [node for i, node in enumerate(self.y_ID) if self.S[i, :].sum() == 1]
            self.nodes = self.build_treestr_from_S(self.S, self.y_ID)
            self.leaves = [node for node in self.nodes if node.type == 'leaf']
            self.root = [node for node in self.nodes if node.type == 'root'][0]
            self.k_level_map = self.build_k_level_map_from_S()
            self.K = np.max(list(self.k_level_map.keys()))
            self.k_level_y = np.sum(self.S, axis=1)

        else:
            raise ValueError("The input structural information is either incorrect or incompatible with the described "
                             "tree dimension.")

    def print_flat(self):
        for node in self.nodes:
            print('node: ', node.id, node.type, '- parent:', node.parent, '- children:', node.children)

    def print(self):
        #self.recursive_print(self.root)
        self.recursive_print(self.leaves)
    def recursive_print(self, node):
        print('\t' * (self.root.k - node.k) + repr(node.id))
        if node.children is not None:
            for child_id in node.children:
                child = [node for node in self.nodes if node.id == child_id][0]
                self.recursive_print(child)

    def create_k_level_map_from_klvlresamplg(self, k_level_resamplings: dict) -> dict:
        """ Function to create the k_level_map out of a pandas sampling rates per tree k_levels.
        k_level_resamplings - dict - dictionary of input dataframes with leaves IDs as dictionary keys. The dictionary
                                     resampling rates are checked to assure a symmetrical tree.
                                     e.g. k_level_resamplings = {1: '1D', 2: '12H', 3: '4H', 4: '2H'}"""
        # Check that 'time_delta' between 'k_level_resamplings' is coherent
        # this assures the temporal tree to be symmetrical
        time_delta = [pd.Timedelta(delta) for delta in k_level_resamplings.values()]
        assert all(max(time_delta) % delta == pd.Timedelta('0') for delta in time_delta)
        # Obtaining number of nodes per k_level and creating the 'k_level_map' dictionary of node IDs
        # Number of nodes per k_level
        k_nodes_nb = [int(max(time_delta)/delta) for delta in time_delta]
        # Aggregation number per k_level
        k_level_aggr = [int(delta / time_delta[-1]) for delta in time_delta]
        # k_level_aggr = k_nodes_nb.copy()
        # k_level_aggr.reverse()
        # Initializing k_level_map & creating it
        k_level_map = {}
        for k in k_level_resamplings:
            k_level_map[k_level_aggr[k - 1]] = [k_level_resamplings[k] + '_' + str(i) for i in range(k_nodes_nb[k - 1])]
        return k_level_map

    def build_k_level_map_from_S(self):
        """Builds the k_lvl_map dictionary out of the summation matrix and y_ID vector."""
        k_level_map = {sum(self.S[0, :]): [self.y_ID[0]]}
        # Keeping track of the created k_aggr_prevs not to overwrite the dictionary as it is constructed
        k_aggr_prevs = [sum(self.S[0, :])]
        yid_prevs = []
        # Looping over the 'm' columns of the summation matrix
        for j in range(self.m):
            col = self.S[:, j]
            # Identifying related y_ID keys for the considered column
            y_keys = [self.y_ID[idx] for idx, val in enumerate(col) if val != 0]
            y_keys_idx = [idx for idx, val in enumerate(col) if val != 0]
            # Looping over the y_ID keys of the considered column
            for idx, y_id in zip(y_keys_idx[1:], y_keys[1:]):
                # The k_aggregation value is equal to the considered line sum of the summation matrix
                k_aggregation = sum(self.S[idx, :])
                # If this aggregation level has not yet been created, do so
                if k_aggregation not in k_aggr_prevs:
                    k_level_map[k_aggregation] = [y_id]
                # If this aggregation level has been created and y_id does not belong to its values, append the list
                elif y_id not in yid_prevs:
                    k_level_map[k_aggregation].append(y_id)
                # Updating lists of dictionary keys and values
                yid_prevs.append(y_id)
                k_aggr_prevs.append(k_aggregation)
        return k_level_map

    def build_S_from_treestr(self, treestr: set):
        """Builds the summation matrix as function of the tree structure attribute."""

        # Defining self.y_ID and its mapping of values
        node_list = list(self.nodes)
        node_list.sort()
        y_ID = [node.id for node in node_list]

        S = np.zeros([self.n, self.m])
        for i, y_id in enumerate(y_ID):
            node = [node for node in treestr if node.id == y_id][0]

            if node.type == 'root':
                S[i, :] = [1]*self.m
            elif node.type == 'node':
                node_leaves = self.identify_leaves_from_node(node, leaves=[])
                S[i, :] = [int(leaf in node_leaves) for leaf in self.leaves_label]
            elif node.type == 'leaf':
                S[i, :] = [int(leaf == node.id) for leaf in self.leaves_label]

        # Re-adjust leaves_label ordering in function of defined y_ID
        self.leaves_label = [node for i, node in enumerate(y_ID) if S[i, :].sum() == 1]
        return S.astype(int), y_ID

    def identify_leaves_from_node(self,
                                  ref_node: Node,
                                  leaves: list = []):
        """Identify tree leaves of a given reference node"""    # 确定叶节点的ID
        if ref_node.type == 'leaf':
            return [ref_node.id]
        else:
            child_nodes = [node for node in self.nodes if node.id in ref_node.children]
            for node in child_nodes:
                if node.type == 'leaf':
                    leaves.append(node.id)
                else:
                    self.identify_leaves_from_node(node, leaves)
            return leaves

    def identify_k_values_of_nodes_from_nodes(self):
        """Loops over tree nodes to fix their k values in function of available self.nodes information"""
        for node in self.nodes:
            leaves = self.identify_leaves_from_node(node, leaves=[])
            node.k = len(leaves)

    def build_treestr_from_S(self,
                             S: np.array,
                             y_ID: list):
        """Builds the tree structure and its attributes from the summation matrix and y_ID vector."""

        structure = self.build_treestr_children_relationships(S, y_ID)
        structure = self.build_treestr_parent_relationships(structure)
        return structure

    def build_treestr_children_relationships(self,
                                             S: np.array,
                                             y_ID: list):
        structure = set()
        tree_leaves = [y_ID[i] for i, val in enumerate(y_ID) if sum(S[i, :]) == 1]
        for i, y_id in enumerate(y_ID):
            S_line = S[i, :]
            k = int(sum(S[i, :]))

            if k == 1:
                node = Node(id=y_id,
                            parent=None,
                            children=None,
                            k=k,
                            type='leaf')
            elif k == np.shape(S)[1]:
                node = Node(id=y_id,
                            parent=None,
                            children=tree_leaves,
                            k=k,
                            type='root')
            else:
                node_leaves = [tree_leaves[idx] for idx, val in enumerate(S_line) if val == 1]
                node = Node(id=y_id,
                            parent=None,
                            children=node_leaves,
                            k=k,
                            type='node')
            structure.add(node)

        return structure

    def identify_next_nodes_to_build(self, nodes, k_lvl_being_built):
        """This function identifies the next_nodes_to_build by increasing the input k_lvl considered.
        It updates the k_lvl_being_built and pool_of_nodes_left_to_build."""
        k_lvl_being_built = min([node.k for node in nodes if node.k > k_lvl_being_built])
        next_nodes_to_build = [node for node in nodes if node.k == k_lvl_being_built]
        next_nodes_to_build = set(next_nodes_to_build)
        pool_of_nodes_left_to_build = set(nodes) - next_nodes_to_build
        return pool_of_nodes_left_to_build, next_nodes_to_build, k_lvl_being_built

    def build_treestr_parent_relationships(self, structure: set):
        leaves = [node for node in structure if node.type == 'leaf']
        nodes = [node for node in structure if node.type != 'leaf']
        k_lvl_being_built = 1

        # Build leaves parent relationship
        for leaf in leaves:
            all_possible_parents = [node for node in nodes if leaf.id in node.children]
            parent_node = self.identify_parent_node(all_possible_parents)
            leaf.parent = [parent.id for parent in parent_node]

        pool_of_nodes_left_to_build, next_nodes_to_build, k_lvl_being_built = self.identify_next_nodes_to_build(nodes, k_lvl_being_built)

        # Iteratively build parent and children relationships over tree nodes
        self.recursive_build_treestr_parents(structure, pool_of_nodes_left_to_build, next_nodes_to_build,
                                             k_lvl_being_built)

        # Update root children relationships
        root = [node for node in structure if node.type == 'root'][0]
        node_children = [node.id for node in structure if node.parent is not None and root.id in node.parent]
        root.children = node_children

        return structure

    def recursive_build_treestr_parents(self, structure, nodes_in, next_nodes_to_build, k_lvl):
        next_nodes = []

        for node_to_link in next_nodes_to_build:
            # Build parent relationship
            all_possible_parents = [node for node in nodes_in if set(node_to_link.children).issubset(node.children)]
            parent_node = self.identify_parent_node(all_possible_parents)
            node_to_link.parent = [parent.id for parent in parent_node]
            next_nodes = next_nodes + parent_node

            # Update children relationship
            node_children = [node.id for node in structure if
                             node.parent is not None and node_to_link.id in node.parent]
            node_to_link.children = node_children

        # Update nodes_left_to_build
        pool_of_nodes_left_to_build, next_nodes, k_lvl = self.identify_next_nodes_to_build(nodes_in, k_lvl)

        if len(pool_of_nodes_left_to_build) == 0:
            return
        else:
            self.recursive_build_treestr_parents(structure, pool_of_nodes_left_to_build, next_nodes, k_lvl)

    def identify_parent_node(self, all_possible_parents):
        """Identifies the parent node from all possible parents through the smallest k_lvl.
        The function encapsulates special cases when multi-parents can be identified, i.e., for spatiotemporal
        dimensions."""

        first_smallest_k = min([node.k for node in all_possible_parents])
        parent_node = [node for node in all_possible_parents if node.k == first_smallest_k]

        if len(parent_node) == 1 and self.dimension == 'spatiotemporal':
            second_smallest_ks = [node.k for node in all_possible_parents
                                  if node.k != first_smallest_k and
                                  node.k < max([node.k for node in all_possible_parents])]
            if len(second_smallest_ks) == 0:
                parent_node = [node for node in all_possible_parents if node.k == first_smallest_k]
            else:
                parent_node = [node for node in all_possible_parents if node.k <= min(second_smallest_ks)]
        return parent_node

    def build_treestr_from_kmap(self, k_level_map) -> set:
        """Function to create tree structure out of the 'k_level_map' dictionary, assuming a symmetrical tree."""
        nodes = set()

        all_k = np.sort(list(k_level_map.keys()), )[::-1]
        for i, k in enumerate(all_k):
            # Build root node
            if k == np.max(list(k_level_map.keys())):
                root_node = Node(id=k_level_map[k][0],
                                 parent=None,
                                 children=k_level_map[all_k[i+1]],
                                 k=k,
                                 type='root')
                nodes.add(root_node)

            # Build leaves
            elif k == 1:
                leaves_to_build = k_level_map[k]

                for leaf_to_build in leaves_to_build:
                    # Identify parent node
                    parent = [node.id for node in nodes if node.type != 'leaf' and leaf_to_build in node.children]
                    # Build node
                    node = Node(id=leaf_to_build,
                                parent=parent,
                                children=None,
                                k=k,
                                type='leaf')
                    nodes.add(node)

            # Build nodes
            else:
                nodes_to_build = k_level_map[k]
                # Identify children subsets
                children_subsets = self.identify_children_subsets_from_klvl(k_level_map, k, all_k[i+1])

                for j, node_to_build in enumerate(nodes_to_build):
                    # Identify parent node
                    parent = [node.id for node in nodes if node_to_build in node.children]
                    # Identify children subsets
                    children = children_subsets[j]
                    # Build node
                    node = Node(id=node_to_build,
                                parent=parent,
                                children=children,
                                k=k,
                                type='node')
                    nodes.add(node)
        return nodes

    def identify_children_subsets_from_klvl(self, k_level_map, k, k_lower) -> list:
        """Function to split the k_lower k_lvl nodes from k_level_map into equally sized chunks of children nodes."""
        children_length = int(len(k_level_map[k_lower]) / len(k_level_map[k]))
        nb_of_children_subsets = int(len(k_level_map[k_lower]) / children_length)
        value_start, value_end = 0, 0
        children = []
        for i in range(nb_of_children_subsets):
            value_end += children_length
            children.append(k_level_map[k_lower][value_start:value_end])
            value_start += children_length
        return children


class Tree(TreeInitialization):
    """A Tree class object to encapsulate hierarchical time-series manipulation functions."""

    def __init__(self, structural_input_information: set or dict or tuple, dimension: str):
        super().__init__(structural_input_information, dimension)

    def create_spatial_hierarchy(self, leaves_dict_df: dict, columns2aggr: list = None) -> None:
        """Creates dictionary storing hierarchical dataframes per node_ID (key).
        Upper level nodes are the result of lower leaves aggregation.

        leaves_dict_df - dict - dictionary of input dataframes with leaves IDs as dictionary keys.
        columns2aggr   - list - default value, i.e., None, aggregates all columns of the input dataframe. Only considers
                                 the input list otherwise."""
        df = {}
        # Define the bottom-layer of the tree
        for leaf in self.leaves_label:
            df[leaf] = leaves_dict_df[leaf]
        # Define nodes to aggregate bottom layer from
        nodes = [node for node in self.nodes if node.type != 'leaf']   # 其中包含了 self.nodes 中所有类型不是 'leaf' 的节点
        # Defining columns and index to consider
        self.hcolumns = df[leaf].columns if columns2aggr is None else columns2aggr      # hierarchical columns
                                # 如果 columns2aggr 为 None，则将 df[leaf] 的列赋值给 self.hcolumns。
                                # 如果 columns2aggr 不为 None，则将 columns2aggr 的值赋值给 self.hcolumns。

        self.lcolumns = df[leaf].columns                                                # leaves columns
        self.index = leaves_dict_df[leaf].index

        # Filling the data hierarchy from bottom level-up
        for node in nodes:
            # Identifying leaf IDs to aggregate per node
            bottom_leaves = self.identify_leaves_from_node(node, leaves=[])
            # print(bottom_leaves)   [D,B,C]
            # Aggregate bottom leaves
            df_aggr = pd.DataFrame(0, index=self.index, columns=self.hcolumns)
            for leaf in bottom_leaves:
                df_aggr[self.hcolumns] = df_aggr[self.hcolumns] + leaves_dict_df[leaf][self.hcolumns]
            # Define node value as the aggregated sum of its bottom leaves  层级的意思，将他们累加
            df[node.id] = df_aggr
        self.df = df


    def create_temporal_hierarchy(self, df_in: pd.DataFrame, columns2aggr: list = None) -> None:
        """Creates dictionary storing hierarchical dataframes per resampling ID (key).
        Upper level nodes are at the resampling rate defined by 'k_level_resamplings'.

        df_in               - pd.DataFrame - input pandas dataframe object to create hierarchy out of.
        columns2aggr        - list         - default value, i.e., None, aggregates all columns of the input dataframe.
                                             Only considers the input list otherwise."""

        self.hcolumns = df_in.columns if columns2aggr is None else columns2aggr
        self.lcolumns = df_in.columns

        df = {}
        for ki, sample in self.k_level_resamplings.items():
            # Obtaining the k level aggregation number
            k = list(self.k_level_map.keys())[ki - 1]
            # Resampling the input dataframe according to the defined resampling rates of 'k_level_resamplings'
            # selected columns are function of the k_level: if k=1 then we consider the leaves_columns (lcolumns),
            # otherwise, hierarchical_columns (hcolumns).
            columns = self.lcolumns if k == 1 else self.hcolumns
            df_resampled = df_in[columns].resample(sample).sum()
            for node_id in self.k_level_map[k]:
                # Attributing dataframe to its node_id key, with an iterative time shift 's'
                s = int(node_id.split('_')[1])
                # that drops the first and last values by 'len(k_level)-s' and 's' respectively due to the 's' shift
                df_resampled_shifted = df_resampled.drop(df_resampled.tail(len(self.k_level_map[k]) - s).index).drop(
                    df_resampled.head(s).index)
                # Selection only rows every 'delta' - where delta is equal to the length of that k_level aggregation
                delta = len(self.k_level_map[k])
                df[node_id] = df_resampled_shifted.loc[::delta, :]
        self.index = df[self.root.id].index
        self.df = df

    def df_klvl(self, k: int):
        """obtain the full dataframe of a k_lvl aggregation from a temporal tree."""
        if self.dimension != 'temporal':
            raise ValueError("There is no temporal dimension in considered tree. This function will not return the "
                             "intended output")
        else:
            df_klvl = pd.DataFrame()
            for n in self.k_level_map[k]:
                df_klvl = pd.concat([df_klvl, self.df[n]])
            return df_klvl.sort_index()

    def df_leaves(self, col: str) -> pd.DataFrame:
        """obtain an aggregated dataframe of a common column of the tree leaves from a spatial tree."""
        if self.dimension == 'spatial':
            df_leaves = pd.DataFrame()
            for leaf in self.leaves_label:
                df_leaves[leaf] = self.df[leaf][col]
            return df_leaves
        else:
            raise ValueError("This tree's dimension is not spatial. The function will not return the intended output")

