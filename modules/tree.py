import pandas as pd
import numpy as np

import helper

class Node:
    def __init__(self, label: str, metric: dict|None=None, split_val: float|int|None =None):
        self.label = label
        self.branches = dict()
        self.metric = metric
        self.split_val = split_val

    def add_branch(self, label: str, node):
        # assert label not in self.branches.keys(), f"Label {label} already exist in this node branches"

        self.branches[label] = node

class RegressionTree:
    def __init__(self, df: pd.DataFrame, metric: str="SSE", min_sample_per_leaf: int=12, max_height: int=3):
        assert len(df) > min_sample_per_leaf
        assert metric.upper() in ["SSE", "MSE", "MAE"]

        # data
        self.df = df
        self.feature_names = self.df.columns.to_list()[:-1]
        self.__feature_indices = {v: k for (k, v) in zip(range(len(self.feature_names)), self.feature_names)}

        self.max_height = max_height
        self.root_node = None
        self.sample_threshold = min_sample_per_leaf
        self.metric = getattr(helper, metric.upper())


    def __choose_subtree_root(self, df):
        root_candidates = df.columns.to_list()[:-1] # get possible keys
        errors_list = []

        for key in root_candidates:
            errors_list.append(self.__metrics_for_key(df, key))

        min_error_idx = np.argmin(np.array(errors_list)[:,1])

        return {"candidate_key": root_candidates[min_error_idx],
                "split_val": errors_list[min_error_idx][0],
                "magnitude": errors_list[min_error_idx][1]}
    

    def __metrics_for_key(self, df, key):
        '''Calculate metrics for given key (column)'''

        metric_list = []
        sorted_df = df.sort_values(key, axis=0)
        
        unique_splits = sorted_df[key].unique()
        split_val_list = np.mean([unique_splits[1:], unique_splits[:-1]], axis=0)

        for split_val in split_val_list:
            split_mask = sorted_df[key] >= split_val # split condition for dataframe

            s1 = sorted_df[split_mask].iloc(axis=1)[-1]
            s2 = sorted_df[~split_mask].iloc(axis=1)[-1]

            error_s1 = self.metric(s1)
            error_s2 = self.metric(s2)

            metric_list.append(error_s1 + error_s2)

        return np.round(split_val_list[np.argmin(metric_list)], decimals=1), np.min(metric_list)
    
    def __get_child_DataFrame(self, df, key, split_val, branch):
        if branch==">=":
            return df[df[key] >= split_val].loc(axis=1)[df.columns != key]
        else:
            return df[df[key] < split_val].loc(axis=1)[df.columns != key]
        
    def __queue_init(self):
        candidate_dict = self.__choose_subtree_root(self.df)
        root_node = Node(label=candidate_dict['candidate_key'], metric=candidate_dict['magnitude'], split_val=candidate_dict['split_val'])

        root_queue_element = {
            'parent_node': root_node,
            'branches': [">=", "<"],
            'df': self.df
        }

        return root_queue_element
        
    def build_tree(self):
        queue = [self.__queue_init()]
        root_node = queue[0]['parent_node']
        self.root_node = root_node
        
        while len(queue) > 0:
            queue_ele = queue.pop(0)

            parent_df = queue_ele['df']
            parent_label = queue_ele['parent_node'].label
            parent_split_val = queue_ele['parent_node'].split_val

            for branch in queue_ele['branches']:
                curr_df = self.__get_child_DataFrame(df=parent_df,key=parent_label, split_val=parent_split_val, branch=branch)

                if len(curr_df)<=self.sample_threshold or len(curr_df.columns) == 1: #TODO
                    leaf_node_val = np.round(np.mean(curr_df.iloc(axis=1)[-1]), decimals=1)
                    leaf_node = Node(label=leaf_node_val)
                    queue_ele['parent_node'].add_branch(branch, leaf_node)
                    continue
                
                new_candidate_dict = self.__choose_subtree_root(curr_df)
                new_node = Node(label=new_candidate_dict['candidate_key'], metric=new_candidate_dict['magnitude'], split_val=new_candidate_dict['split_val'])
                queue_ele['parent_node'].add_branch(branch, new_node)

                # if new_candidate_dict['magnitude'] < 1e-3:
            
                new_queue_element = {}
                new_queue_element['parent_node'] = new_node
                new_queue_element['branches'] = [">=", "<"]
                new_queue_element['df'] = curr_df

                queue.append(new_queue_element)

    def predict(self, features: np.array):
        assert self.root_node != None, "Tree has not been built, run build_tree() first."
        assert isinstance(features, np.ndarray), "Inputs must be numpy array"
        assert len(features.shape) == 2
        assert len(self.__feature_indices) == features.shape[1]

        curr_node = self.root_node
        ret = []

        for feature in features:
            while curr_node.metric != None:
                key = self.__feature_indices[curr_node.label]

                if feature[key] >= curr_node.split_val:
                    curr_node = curr_node.branches['>=']
                else:
                    curr_node = curr_node.branches['<']

            ret.append(curr_node.label)
            curr_node = self.root_node

        return np.array(ret)
    
    def validate(self, df: pd.DataFrame, metric="MAE"):
        features = df.iloc(axis=1)[:-1].to_numpy()
        targets = df.iloc(axis=1)[-1].to_numpy()

        outputs = self.predict(features=features)

        return getattr(helper, metric)(outputs, targets, True)