# vim: set textwidth=0:
from torch.utils.data import Dataset
import torch
import json
import os
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def preprocess_computation(data_file: str, add_type: bool, pre_com: bool):
    '''
    Args:
        data_file: path to the data file
        add_type: whether to add logic type information
        pre_com: whether to add pre-computed information.
    '''
    with open(data_file, 'r') as f:
        data = json.load(f)
        new_data = []
        for d in tqdm(data):
            table_header = d["table_header"]
            table_cont = d["table_cont"]
            h_idx = d['highlight_cells']
            src_text = "<table> " + "<caption> " +  d['topic'] + " </caption> "
            if add_type:
                src_text = "<type> " + d['action'] + "</type> " + src_text
            # Construct pandas table
            pd_in = defaultdict(list)
            for ind, header in enumerate(table_header):
                for inr, row in enumerate(table_cont):
                    if inr == len(table_cont) - 1 \
                            and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
                                 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                        continue
                    pd_in[header].append(row[ind])
            pd_table = pd.DataFrame(pd_in)
            # precomputed ranks
            for row, col, max_rank, min_rank in h_idx:
                val = pd_table[col].iloc[row]
                if pre_com and max_rank is not None:
                    cell_str = f"<cell> {val} <col_header> {col} </col_header> <row_idx> {row} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank></cell> "
                else:
                    cell_str = f"<cell> {val} <col_header> {col} </col_header> <row_idx> {row} </row_idx> </cell> "
                src_text += cell_str
            # precomputed aggregation values
            if pre_com and d['agg_cells']:
                sum_cell, avg_cell = d['agg_cells']
                src_text += f"<sum_cell> {sum_cell[1]} <col_header> {sum_cell[0]} </col_header> </sum_cell> "
                src_text += f"<avg_cell> {avg_cell[1]} <col_header> {avg_cell[0]} </col_header> </avg_cell> "
            src_text += "</table>"
            d['src_text'] = src_text
            new_data.append(d)
        return new_data


class ContlogDataset(Dataset):

    def __init__(self, data_file, tokenizer, max_src_len, max_tgt_len, task, add_type=False, pre_com=True):
        self.data = preprocess_computation(data_file, add_type, pre_com)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        src_text = d['src_text'].strip()
        src_text = ' '.join(src_text.split())

        if self.task == 'text' and 'sent' in d:
            tgt_text = d['sent'].strip()
        elif self.task == 'logic' and 'logic_str' in d:
            tgt_text = d['logic_str'].strip()
        else:
            raise NotImplementedError

        tgt_text = ' '.join(tgt_text.split())

        source = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_len
        )
        target = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tgt_len
        )

        source_input_ids = torch.LongTensor(source.data["input_ids"])
        target_input_ids = torch.LongTensor(target.data["input_ids"])
        source_mask = torch.LongTensor(source.data["attention_mask"])
        return {
            'source_ids': source_input_ids,
            'source_mask': source_mask,
            'target_ids': target_input_ids
        }

class ArrayStack():
    """LIFO Stack implementation using a Python list as underlying storage"""

    def __init__(self, n):
        """Create an empty stack."""
        self.data = []
        self.maxLen = n  # n : an integer that represent the max elements capacity of the stack

    def __len__(self):
        """Return the number of elements in the stack"""
        return len(self.data)

    def is_empty(self):
        """Return True if the stack is empty"""
        return len(self.data) == 0

    def is_full(self):
        """Return True if the stack is full"""
        return len(self.data) == self.maxLen

    def push(self, e):
        """Add element e to the top of the stack

         Raise Empty exception if the stack is full"""
        if self.is_full():
            raise AssertionError('Stack is full')
        return self.data.append(e)

    def top(self):
        """Return the element at the top of the stack, but not move it.

        Raise Empty exception if the stack is empty"""
        if self.is_empty():
            raise AssertionError('Stack is empty')
        return self.data[-1]

    def pop(self):
        """Return the element at the top of the stack, meanwhile move it.

        Raise Empty exception if the stack is empty"""
        if self.is_empty():
            raise AssertionError('Stack is empty')
        return self.data.pop()




def str2list(logic_str : str) -> list:
    '''

    :param logic_str: original logic_str
    :return: splitted token list
    '''

    # Prune affix "= true" if it exists
    if logic_str.endswith('true'):
        logic_str = logic_str[:-7]
    # detect empty column headers
    while logic_str.find("; }") > 0:
        idx = logic_str.find("; }")
        logic_str = logic_str[:idx+2] + "[None] " + logic_str[idx+2:]
    while logic_str.find("; ;") > 0:
        idx = logic_str.find("; ;")
        logic_str = logic_str[:idx+2] + "[None] " + logic_str[idx+2:]
    unreplaced_logic = logic_str[:].split(" ")
    logic = []
    for tok in unreplaced_logic:
        if tok == "[None]":
            tok = ""
        logic.append(tok)
    token_list = []
    i = 0
    while i < len(logic):
        cur_token = logic[i]
        if cur_token in ["{", "}", ";"]:
            token_list.append(cur_token)
            i = i + 1
            continue
        i = i + 1
        while i < len(logic) and not logic[i] in ["{", "}", ";"]:
            cur_token = " ".join([cur_token, logic[i]])
            i = i + 1
        token_list.append(cur_token)
    return token_list

def parse_str(logic_str : str, func_map):
    '''
    Parsing a logical form from a logic str
    Args:
        logic_str: a logic str
        func_map: a function-to-function map

    Returns:
        final_form: a structured logical form, dict
    '''
    token_list = str2list(logic_str)
    logic_stack = ArrayStack(len(token_list))
    func_stack = []
    i = 0
    func_idx = 0
    while i < len(token_list):
        cur_dict = {}
        cur_args = []
        while token_list[i] != "}":
            logic_stack.push(token_list[i])
            i = i + 1
        while logic_stack.top() != "{":
            if logic_stack.top() != ";" and isinstance(logic_stack.top(), str):
                cur_args.append(logic_stack.pop())
            elif logic_stack.top() == ";":
                logic_stack.pop()
            elif isinstance(logic_stack.top(), int):
                cur_args.append(func_stack[logic_stack.pop()])
        # pop "{"
        logic_stack.pop()
        # pop and store the function
        func = logic_stack.pop()
        if func in func_map.keys():
            func = func_map[func]
        cur_dict["func"] = func
        cur_dict["args"] = cur_args[::-1]
        func_stack.append(cur_dict)
        # push the index into logic_stack
        logic_stack.push(func_idx)
        func_idx += 1
        i = i + 1
    final_form = func_stack[-1]
    return final_form



import re
import math
import pandas as pd
import numpy as np
import datetime

APIs = {}


# With only one argument

### count
APIs['count'] = {"argument":['row'], 'output': 'num', 
                 'function': lambda t :  len(t),
                 'tostr': lambda t : "count {{ {} }}".format(t),
                 'append': True}

### unique
APIs['only'] = {"argument":['row'], 'output': 'bool',
                "function": lambda t: len(t) == 1,
                "tostr": lambda t : "only {{ {} }}".format(t),
                'append': None}


# With only two argument and the first is row
APIs['str_hop'] = {"argument":['row', 'header'], 'output': 'str', 
               'function': lambda t, col :  hop_op(t, col),
               'tostr': lambda t, col : "str_hop {{ {} ; {} }}".format(t, col),
               'append': True}

APIs['num_hop'] = {"argument":['row', 'header'], 'output': 'obj', 
               'function': lambda t, col :  hop_op(t, col),
               'tostr': lambda t, col : "num_hop {{ {} ; {} }}".format(t, col),
               'append': True}

APIs['avg'] = {"argument":['row', 'header'], 'output': 'num',
              "function": lambda t, col : agg(t, col, "mean"),
              "tostr": lambda t, col : "avg {{ {} ; {} }}".format(t, col),
              'append': True}

APIs['sum'] = {"argument":['row', 'header'], 'output': 'num',
              "function": lambda t, col : agg(t, col, "sum"),
              "tostr": lambda t, col : "sum {{ {} ; {} }}".format(t, col),
              'append': True}

APIs['max'] = {"argument":['row', 'header'], 'output': 'obj',
              "function": lambda t, col : nth_maxmin(t, col, order=1, max_or_min="max", arg=False),
              "tostr": lambda t, col : "max {{ {} ; {} }}".format(t, col),
              'append': True}

APIs['min'] = {"argument":['row', 'header'], 'output': 'obj',
                "function": lambda t, col : nth_maxmin(t, col, order=1, max_or_min="min", arg=False),
                "tostr": lambda t, col : "min {{ {} ; {} }}".format(t, col),
                'append': True}

APIs['argmax'] = {"argument":['row', 'header'], 'output': 'row',
                  'function': lambda t, col : nth_maxmin(t, col, order=1, max_or_min="max", arg=True),
                  'tostr': lambda t, col : "argmax {{ {} ; {} }}".format(t, col),
                  'append': False}

APIs['argmin'] = {"argument":['row', 'header'], 'output': 'row',
                  'function': lambda t, col :  nth_maxmin(t, col, order=1, max_or_min="min", arg=True),
                  'tostr': lambda t, col : "argmin {{ {} ; {} }}".format(t, col),
                  'append': False}


# add for ordinal
APIs['nth_argmax'] = {"argument":['row', 'header', 'num'], 'output': 'row',
                  'function': lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="max", arg=True),
                  'tostr': lambda t, col, ind : "nth_argmax {{ {} ; {} ; {} }}".format(t, col, ind),
                  'append': False}

APIs['nth_argmin'] = {"argument":['row', 'header', 'num'], 'output': 'row',
                  'function': lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="min", arg=True),
                  'tostr': lambda t, col, ind : "nth_argmin {{ {} ; {} ; {} }}".format(t, col, ind),
                  'append': False}

APIs['nth_max'] = {"argument":['row', 'header', 'num'], 'output': 'num',
              "function": lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="max", arg=False),
              "tostr": lambda t, col, ind : "nth_max {{ {} ; {} ; {} }}".format(t, col, ind),
              'append': True}

APIs['nth_min'] = {"argument":['row', 'header', 'num'], 'output': 'num',
                "function": lambda t, col, ind : nth_maxmin(t, col, order=ind, max_or_min="min", arg=False),
                "tostr": lambda t, col, ind : "nth_min {{ {} ; {} ; {} }}".format(t, col, ind),
                'append': True}



# With only two argument and the first is not row
APIs['diff'] = {"argument":['obj', 'obj'], 'output': 'str', 
                'function': lambda t1, t2 : obj_compare(t1, t2, type="diff"),
                'tostr': lambda t1, t2 : "diff {{ {} ; {} }}".format(t1, t2),
                'append': True}

APIs['greater'] = {"argument":['obj', 'obj'], 'output': 'bool', 
                   'function': lambda t1, t2 :  obj_compare(t1, t2, type="greater"),
                   'tostr': lambda t1, t2 : "greater {{ {} ; {} }}".format(t1, t2),
                   'append': False}

APIs['less'] = {"argument":['obj', 'obj'], 'output': 'bool', 
                'function': lambda t1, t2 :  obj_compare(t1, t2, type="less"),
                'tostr': lambda t1, t2 : "less {{ {} ; {} }}".format(t1, t2),
                'append': True}

APIs['eq'] = {"argument":['obj', 'obj'], 'output': 'bool', 
              'function': lambda t1, t2 :  obj_compare(t1, t2, type="eq"),
              'tostr': lambda t1, t2 : "eq {{ {} ; {} }}".format(t1, t2),
              'append': None}

APIs['not_eq'] = {"argument":['obj', 'obj'], 'output': 'bool', 
                 'function': lambda t1, t2 : obj_compare(t1, t2, type="not_eq"),
                 'tostr': lambda t1, t2 : "not_eq {{ {} ; {} }}".format(t1, t2),
                 "append": None}

APIs['str_eq'] = {"argument":['str', 'str'], 'output': 'bool',
                  # 'function': lambda t1, t2 :  obj_compare(t1, t2, type="eq"),
                  'function': lambda t1, t2 :  t1 in t2 or t2 in t1,
                  'tostr': lambda t1, t2 : "str_eq {{ {} ; {} }}".format(t1, t2),
                  "append": None}

APIs['not_str_eq'] = {"argument":['str', 'str'], 'output': 'bool',
                     # 'function': lambda t1, t2 :  obj_compare(t1, t2, type="not_eq"),
                     'function': lambda t1, t2 :  t1 not in t2 and t2 not in t1,
                     'tostr': lambda t1, t2 : "not_str_eq {{ {} ; {} }}".format(t1, t2),
                     "append": None}

APIs['round_eq'] = {"argument":['obj', 'obj'], 'output': 'bool', 
              'function': lambda t1, t2 :  obj_compare(t1, t2, round=True, type="eq"),
              'tostr': lambda t1, t2 : "round_eq {{ {} ; {} }}".format(t1, t2),
              'append': None}

APIs['and'] = {"argument":['bool', 'bool'], 'output': 'bool',
                'function': lambda t1, t2 :  t1 and t2,
                'tostr': lambda t1, t2 : "and {{ {} ; {} }}".format(t1, t2),
                "append": None}



# With only three argument and the first is row
# str
APIs["filter_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "row", 
                        "function": lambda t, col, value: fuzzy_match_filter(t, col, value),
                        "tostr":lambda t, col, value: "filter_str_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        'append': False}

APIs["filter_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "row", 
                        "function": lambda t, col, value: fuzzy_match_filter(t, col, value, negate=True),
                        "tostr":lambda t, col, value: "filter_str_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        'append': False}

# obj: num or str
APIs["filter_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row", 
                    "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="eq"),
                    "tostr":lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                    'append': False}

APIs["filter_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row", 
                    "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="not_eq"),
                    "tostr":lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                    'append': False}

APIs["filter_less"] = {"argument": ['row', 'header', 'obj'], "output": "row", 
                        "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less"),
                        "tostr":lambda t, col, value: "filter_less {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": False}

APIs["filter_greater"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                        "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater"),
                        "tostr":lambda t, col, value: "filter_greater {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": False}

APIs["filter_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                             "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater_eq"),
                             "tostr":lambda t, col, value: "filter_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                             "append": False}

APIs["filter_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                          "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less_eq"),
                          "tostr":lambda t, col, value: "filter_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": False}

APIs["filter_all"] = {"argument": ['row', 'header'], "output": "row", 
                        "function": lambda t, col: (t, col),
                        "tostr":lambda t, col: "filter_all {{ {} ; {} }}".format(t, col),
                        'append': False}




# all
# str
APIs["all_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                        "function": lambda t, col, value: all_str_filter(t, col, value),
                        "tostr":lambda t, col, value: "all_str_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["all_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                  "function": lambda t, col, value: all_str_filter(t, col, value, neg=True),
                  "tostr":lambda t, col, value: "all_str_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

# obj: num or str
APIs["all_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: all_filter(t, col, value, type="eq"),
                  "tostr":lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["all_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: all_filter(t, col, value, type="eq",neg=True),
                  "tostr":lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["all_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                    "function": lambda t, col, value: all_filter(t, col, value, type="less"),
                    "tostr":lambda t, col, value: "all_less {{ {} ; {} ; {} }}".format(t, col, value),
                    "append": None}

APIs["all_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: all_filter(t, col, value, type="less_eq"),
                        "tostr":lambda t, col, value: "all_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["all_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: all_filter(t, col, value, type="greater"),
                       "tostr":lambda t, col, value: "all_greater {{ {} ; {} ; {} }}".format(t, col, value),
                       "append": None}

APIs["all_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: all_filter(t, col, value, type="greater_eq"),
                          "tostr":lambda t, col, value: "all_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}



# most
# str
APIs["most_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                        "function": lambda t, col, value: most_str_filter(t, col, value, neg=False),
                        "tostr":lambda t, col, value: "most_str_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["most_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
                  "function": lambda t, col, value: most_str_filter(t, col, value, neg=True),
                  "tostr":lambda t, col, value: "most_str_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

# obj: num or str
APIs["most_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: most_filter(t, col, value, type="eq", neg=False),
                  "tostr":lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["most_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                  "function": lambda t, col, value: most_filter(t, col, value, type="eq", neg=True),
                  "tostr":lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                  "append": None}

APIs["most_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                    "function": lambda t, col, value: most_filter(t, col, value, type="less"),
                    "tostr":lambda t, col, value: "most_less {{ {} ; {} ; {} }}".format(t, col, value),
                    "append": None}

APIs["most_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: most_filter(t, col, value, type="less_eq"),
                        "tostr":lambda t, col, value: "most_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

APIs["most_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: most_filter(t, col, value, type="greater"),
                       "tostr":lambda t, col, value: "most_greater {{ {} ; {} ; {} }}".format(t, col, value),
                       "append": None}

APIs["most_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: most_filter(t, col, value, type="greater_eq"),
                          "tostr":lambda t, col, value: "most_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}




month_map = {'january': 1, 'february':2, 'march':3, 'april':4, 'may':5, 'june':6,
                 'july':7, 'august':8, 'september':9, 'october':10, 'november':11, 'december':12, 
                 'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

### regex list

# number format: 
'''
10
1.12
1,000,000
10:00
1st, 2nd, 3rd, 4th
'''
pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

pat_add = r"((?<==\s)\d+)"

# dates
pat_year = r"\b(\d\d\d\d)\b"
pat_day = r"\b(\d\d?)\b"
pat_month = r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:rch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b"


class ExeError(object):
  def __init__(self, message="exe error"):
    self.message = message


### for filter functions. we reset index for the result

# filter_str_eq / not_eq
def fuzzy_match_filter(t, col, val, negate=False, return_index=True):

  trim_t = t[col].str.replace(" ", "")
  trim_val = val.replace(" ", "")

  if negate:
    res = t[~trim_t.str.contains(trim_val, regex=False)]
  else:
    res = t[trim_t.str.contains(trim_val, regex=False)]
  row_index = res.index
  # res = res.reset_index(drop=True)
  if return_index:
      return res, row_index, col
  else:
      return res

# filter nums ...
def fuzzy_compare_filter(t, col, val, type, return_index=True):
  '''
  fuzzy compare and filter out rows. 
  return empty pd if invalid

  type: eq, not_eq, greater, greater_eq, less, less_eq
  '''
  t = t.copy()

  t[col] = t[col].astype('str')
  # t.loc[:, col] = t.loc[:, col].astype('str')

  # dates
  if len(re.findall(pat_month, val)) > 0:
    year_list = t[col].str.extract(pat_year, expand=False)
    day_list = t[col].str.extract(pat_day, expand=False)
    month_list = t[col].str.extract(pat_month, expand=False)
    month_num_list = month_list.map(month_map)

    # pandas at most 2262
    year_list = year_list.fillna("2260").astype("int")
    day_list = day_list.fillna("1").astype("int")
    month_num_list = month_num_list.fillna("1").astype("int")

    # print (year_list)
    # print (day_list)
    # print (month_num_list)

    date_frame = pd.to_datetime(pd.DataFrame({'year': year_list,'month':month_num_list,'day':day_list}))
    # print (date_frame)

    # for val
    year_val = re.findall(pat_year, val)
    if len(year_val) == 0:
      year_val = year_list.iloc[0]
    else:
      year_val = int(year_val[0])

    day_val = re.findall(pat_day, val)
    if len(day_val) == 0:
      day_val = day_list.iloc[0]
    else:
      day_val = int(day_val[0])

    month_val = re.findall(pat_month, val)
    if len(month_val) == 0:
      month_val = month_num_list.iloc[0]
    else:
      month_val = month_map[month_val[0]]

    date_val = datetime.datetime(year_val, month_val, day_val)
    # print (date_val)

    if type == "greater":
      res = t[date_frame > date_val]
    elif type == "greater_eq":
      res = t[date_frame >= date_val]
    elif type == "less":
      res = t[date_frame < date_val]
    elif type == "less_eq":
      res = t[date_frame <= date_val]
    elif type == "eq":
      res = t[date_frame == date_val]
    elif type == "not_eq":
      res = t[~date_frame != date_val]
    row_index = res.index

    # res = res.reset_index(drop=True)
    if return_index:
        return res, row_index, col
    else:
        return res



  # numbers, or mixed numbers and strings
  val_pat = re.findall(pat_num, val)
  if len(val_pat) == 0:
    # return pd.DataFrame(columns=list(t.columns))
    # fall back to full string matching

    if type == "eq":
      res = t[t[col].str.contains(val, regex=False)]
      if return_index:
          return res, res.index, col
    elif type == "not_eq":
      res = t[~t[col].str.contains(val, regex=False)]
      if return_index:
          return res, res.index, col
    else:
      return pd.DataFrame(columns=list(t.columns))

    # return pd.DataFrame(columns=list(t.columns))

  num = val_pat[0].replace(",", "")
  num = num.replace(":", "")
  num = num.replace(" ", "")
  try:
    num = float(num)
  except:
    num = num.replace(".", "")
    num = float(num)
  # print (num)

  pats = t[col].str.extract(pat_add, expand=False)
  if pats.isnull().all():
    pats = t[col].str.extract(pat_num, expand=False)
  if pats.isnull().all():
    return pd.DataFrame(columns=list(t.columns))
  nums = pats.str.replace(",", "")
  nums = nums.str.replace(":", "")
  nums = nums.str.replace(" ", "")
  try:
    nums = nums.astype("float")
  except:
    nums = nums.str.replace(".", "")
    nums = nums.astype("float")
  # print (nums)

  if type == "greater":
    res = t[np.greater(nums, num)]
  elif type == "greater_eq":
    res = t[np.greater_equal(nums, num)]
  elif type == "less":
    res = t[np.less(nums, num)]
  elif type == "less_eq":
    res = t[np.less_equal(nums, num)]
  elif type == "eq":
    res = t[np.isclose(nums, num)]
  elif type == "not_eq":
    res = t[~np.isclose(nums, num)]

  row_index = res.index
  '''
    I removed this line so that the real index of selected rows are returned.
    However, a bug happend when performing .iloc[] operations on the filtered rows.(e.g. in nth_max_min())
    To fix the bug, I had two choices:
    (1) uncomment this line (X)
    (2) Keep this line commented. (V)
  '''
  # res = res.reset_index(drop=True)

  if return_index:
      return res, row_index, col
  else:
      return res

  # all invalid

  return pd.DataFrame(columns=list(t.columns))


###for majority- functions, with index returned.
def most_str_filter(t, col, val, neg=False, return_index=True):
    res, row, col = fuzzy_match_filter(t, col, val, return_index=True)
    if neg:
        return len(t) // 3 > len(res),  col
    else:
        return len(t) // 3 <= len(res), col

def most_filter(t, col, val, type, neg=False, return_index=True):
    res, row, col = fuzzy_compare_filter(t, col, val, type, return_index=True)
    if neg:
        return len(t) // 3 > len(res),  col
    else:
        return len(t) // 3 <= len(res), col

def all_str_filter(t, col, val, neg=False, return_index=True):
    res, row, col = fuzzy_match_filter(t, col, val, return_index=True)
    if neg:
        return 0 == len(res),  col
    else:
        return len(t) == len(res), col

def all_filter(t, col, val, type, neg=False, return_index=True):
    res, row, col = fuzzy_compare_filter(t, col, val, type, return_index=True)
    if neg:
        return 0 == len(res),  col
    else:
        return len(t) == len(res), col


### for comparison
def obj_compare(num1, num2, round=False, type="eq"):


  tolerance = 0.15 if round else 1e-9
  # both numeric
  try:
    num_1 = float(num1)
    num_2 = float(num2)

    # if negate:
    #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    # return math.isclose(num_1, num_2, rel_tol=tolerance)

    if type == "eq":
      return math.isclose(num_1, num_2, rel_tol=tolerance)
    elif type == "not_eq":
      return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    elif type == "greater":
      return num_1 > num_2
    elif type == "less":
      return num_1 < num_2
    elif type == "diff":
      return num_1 - num_2




  except ValueError:
    # strings
    # mixed numbers and strings
    num1 = str(num1)
    num2 = str(num2)

    # if type == "eq" and ( num1 in num2 or num2 in num1 ):
    #     return True

    # dates
    # num1
    if len(re.findall(pat_month, num1)) > 0:
      year_val1 = re.findall(pat_year, num1)
      if len(year_val1) == 0:
        year_val1 = int("2260")
      else:
        year_val1 = int(year_val1[0])

      day_val1 = re.findall(pat_day, num1)
      if len(day_val1) == 0:
        day_val1 = int("1")
      else:
        day_val1 = int(day_val1[0])

      month_val1 = re.findall(pat_month, num1)
      if len(month_val1) == 0:
        month_val1 = int("1")
      else:
        month_val1 = month_map[month_val1[0]]

      try:
        date_val1 = datetime.datetime(year_val1, month_val1, day_val1)
      except:
        return ExeError

      # num2
      year_val2 = re.findall(pat_year, num2)
      if len(year_val2) == 0:
        year_val2 = int("2260")
      else:
        year_val2 = int(year_val2[0])

      day_val2 = re.findall(pat_day, num2)
      if len(day_val2) == 0:
        day_val2 = int("1")
      else:
        day_val2 = int(day_val2[0])

      month_val2 = re.findall(pat_month, num2)
      if len(month_val2) == 0:
        month_val2 = int("1")
      else:
        month_val2 = month_map[month_val2[0]]

      try:
        date_val2 = datetime.datetime(year_val2, month_val2, day_val2)
      except:
        return ExeError

      # if negate:
      #   return date_val1 != date_val2
      # else:
      #   return date_val1 == date_val2

      if type == "eq":
        return date_val1 == date_val2
      elif type == "not_eq":
        return date_val1 != date_val2
      elif type == "greater":
        return date_val1 > date_val2
      elif type == "less":
        return date_val1 < date_val2
      # for diff return string
      elif type == "diff":
        return str((date_val1 - date_val2).days) + " days"


    # mixed string and numerical
    val_pat1 = re.findall(pat_num, num1)
    val_pat2 = re.findall(pat_num, num2)
    if len(val_pat1) == 0 or len(val_pat2) == 0:

      # fall back to full string matching
      if type == "not_eq":
        return (num1 not in num2) and (num2 not in num1)
      elif type == "eq":
        return num1 in num2 or num2 in num1
      else:
        return ExeError()

    num_1 = val_pat1[0].replace(",", "")
    num_1 = num_1.replace(":", "")
    num_1 = num_1.replace(" ", "")
    try:
      num_1 = float(num_1)
    except:
      num_1 = num_1.replace(".", "")
      num_1 = float(num_1)

    num_2 = val_pat2[0].replace(",", "")
    num_2 = num_2.replace(":", "")
    num_2 = num_2.replace(" ", "")
    try:
      num_2 = float(num_2)
    except:
      num_2 = num_2.replace(".", "")
      num_2 = float(num_2)


    # if negate:
    #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    # return math.isclose(num_1, num_2, rel_tol=tolerance)

    if type == "eq":
      return math.isclose(num_1, num_2, rel_tol=tolerance)
    elif type == "not_eq":
      return (not math.isclose(num_1, num_2, rel_tol=tolerance))
    elif type == "greater":
      return num_1 > num_2
    elif type == "less":
      return num_1 < num_2
    elif type == "diff":
      return num_1 - num_2




### for aggregation: sum avg

def agg(t, col, type, return_index=True):
  '''
  sum or avg for aggregation
  '''

  # unused
  if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
    if type == "sum":
      res = t[col].sum()
    elif type == "avg":
      res = t[col].mean()

    return res, col if return_index else res

  else:

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
      pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
      return 0.0, col if return_index else 0.0
    pats.fillna("0.0")
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
      nums = nums.astype("float")
    except:
      nums = nums.str.replace(".", "")
      nums = nums.astype("float")

    # print (nums)
    if type == "sum":
      return nums.sum(), col if return_index else nums.sum()
    elif type == "mean":
      return nums.mean(), col if return_index else nums.mean()

def add_agg_cell(t, col):
  '''
  sum or avg for aggregation
  '''

  # unused
  if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
    sum_res = t[col].sum()
    avg_res = t[col].mean()

    return sum_res, avg_res

  else:

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
      pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
      raise ExeError
    pats.fillna("0.0")
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
      nums = nums.astype("float")
    except:
      nums = nums.str.replace(".", "")
      nums = nums.astype("float")

    # print (nums)

    return nums.sum(), nums.mean()

### for hop 

def hop_op(t, col, return_index=True):
  if len(t) == 0:
    return ExeError()

  return t[col].values[0], col



### return processed table to compute ranks of cells
def process_num_table(t, col):
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")


        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            # print (date_series)



            return date_series

        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        raise ExeError()
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    return nums






def nth_maxmin(t, col, order=1, max_or_min="max", arg=False, return_index=True):
    '''
      for max, min, argmax, argmin,
      nth_max, nth_min, nth_argmax, nth_argmin

      return string or rows
      '''

    order = int(order)
    ### return the original content for max,min
    # dates
    date_pats = t[col].str.extract(pat_month, expand=False)
    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        # print (year_list)
        # print (day_list)
        # print (month_num_list)

        try:
            date_series = pd.to_datetime(pd.DataFrame({'year': year_list, 'month': month_num_list, 'day': day_list}))
            # print (date_series)

            if max_or_min == "max":
                tar_row = date_series.nlargest(order).iloc[[-1]]
            elif max_or_min == "min":
                tar_row = date_series.nsmallest(order).iloc[[-1]]
                # order *= -1

            ind = list(tar_row.index.values)
            if arg:
                res = t.loc[ind]
            else:
                res = t.loc[ind][col].values[0]

            if return_index:
                return res, t.index, col, t.index
            else:
                return res

        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return ExeError()
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    try:
        if max_or_min == "max":
            tar_row = nums.nlargest(order).iloc[[-1]]
        elif max_or_min == "min":
            tar_row = nums.nsmallest(order).iloc[[-1]]
        ind = list(tar_row.index.values)
        # print (ind)
        # print (t.iloc[ind][col].values)
        if arg:
            res = t.loc[ind]
        else:
            res = t.loc[ind][col].values[0]

    except:
        return ExeError()

    # print (res)
    if return_index:
        return res, t.index, col, (t.index, order)
    else:
        return res





def is_ascii(s):
    return all(ord(c) < 128 for c in s)



if __name__ == '__main__':
    t = pd.DataFrame([('bird', 389.0),
                       ('bird', 24.0),
                       ('anc', 80.5),
                       ('mammal', np.nan)],
                      columns=('class', 'max_speed'))
    col = 'class'
    val = 'bird'
    trim_t = t[col].str.replace(" ", "")
    trim_val = val.replace(" ", "")



    res = t[trim_t.str.contains(trim_val, regex=False)]

    row = res.nlargest(2, columns="max_speed").iloc[[-1]].index
    print(row)
    # print(res.index[0], res.)






import json
import random
from tqdm import tqdm
from collections import defaultdict
import itertools as it
import pandas as pd

func_map = {
  "hop" : "num_hop"
}

func_map_str_replace = {
        "num_hop": "str_hop",
        "eq": "str_eq",
        "filter_eq": "filter_str_eq",
        "not_eq": "not_str_eq",
        "filter_not_eq": "filter_str_not_eq",
        "all_eq": "all_str_eq",
        "all_not_eq": "all_str_not_eq",
        "most_eq": "most_str_eq",
        'most_not_eq': 'most_str_not_eq'
    }

class Node(object):
    def __init__(self, full_table, dict_in):
        '''
		construct tree
		'''
        self.swap_dict = defaultdict(list)
        for op, attr in APIs.items():
            self.swap_dict[' '.join(attr['argument'])].append(op)

        self.full_table = full_table
        self.func = dict_in["func"]
        self.dict_in = dict_in

        # row, num, str, obj, header, bool
        self.arg_type_list = APIs[self.func]["argument"]
        self.arg_list = []

        # [("text_node", a), ("func_node", b)]
        self.child_list = []
        child_list = dict_in["args"]

        assert len(self.arg_type_list) == len(child_list)

        # bool, num, str, row
        self.out_type = APIs[self.func]["output"]

        for each_child in child_list:
            if isinstance(each_child, str):
                self.child_list.append(("text_node", each_child))
            elif isinstance(each_child, dict):
                sub_func_node = Node(self.full_table, each_child)
                self.child_list.append(("func_node", sub_func_node))
            else:
                raise ValueError("child type error")

        self.result = None

    def eval(self):

        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    self.arg_list.append(self.full_table)
                else:
                    self.arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].eval()
                # print ("exit func: ", each_child[1].func)

                # invalid
                if isinstance(sub_result, ExeError):
                    # print ("sublevel error")
                    return ExeError()
                elif each_type == "row":
                    if not isinstance(sub_result, pd.DataFrame):
                        # print ("error function return type")
                        return ExeError()
                elif each_type == "bool":
                    if not isinstance(sub_result, bool):
                        # print ("error function return type")
                        return ExeError()
                elif each_type == "str":
                    if not isinstance(sub_result, str):
                        # print ("error function return type")
                        return ExeError()

                self.arg_list.append(sub_result)

        result = APIs[self.func]["function"](*self.arg_list)
        return result

    def eval_index(self):
        row, col, row_scope = [], [], []

        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    self.arg_list.append(self.full_table)
                else:
                    self.arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].eval_index()
                if isinstance(sub_result, tuple) and len(sub_result) == 4:
                    sub_result, new_row, new_col, row_scope = sub_result
                    if not isinstance(new_row, list):
                        new_row = new_row.to_list()
                    # if not isinstance(row_scope, list):
                    #     row_scope = row_scope.to_list()
                    # if row_scope:
                    #     scope, order = row_scope
                    #     if not isinstance(scope, list):
                    #         scope = scope.to_list()
                    #     row_scope = (scope, order)


                    col.extend(new_col)
                    row.extend(new_row)


                elif isinstance(sub_result, tuple) and len(sub_result) == 3:
                    sub_result, new_row, new_col = sub_result
                    if not isinstance(new_row, list):
                        new_row = new_row.to_list()

                    col.extend(new_col)
                    row.extend(new_row)
                    # if self.func == 'count' or self.func == 'only':
                    #     return row, col
                elif isinstance(sub_result, tuple) and len(sub_result) == 2:
                    sub_result,  new_col = sub_result
                    col.extend(new_col)
                    # return sub_result



                # print ("exit func: ", each_child[1].func)

                # invalid
                if isinstance(sub_result, ExeError):
                    # print ("sublevel error")
                    return ExeError(), row, col, row_scope
                elif each_type == "row":
                    if not isinstance(sub_result, pd.DataFrame):
                        # print ("error function return type")
                        return ExeError(), row, col, row_scope
                elif each_type == "bool":
                    if not isinstance(sub_result, bool):
                        # print ("error function return type")
                        return ExeError(), row, col, row_scope
                elif each_type == "str":
                    if not isinstance(sub_result, str):
                        # print ("error function return type")
                        return ExeError(), row, col, row_scope

                self.arg_list.append(sub_result)

        result = APIs[self.func]["function"](*self.arg_list)
        if isinstance(result, tuple) and len(result) == 4:
            #row_scope is the subtable rows for functions like max/min, we store it to compute ranks
            new_result, new_row, new_col, row_scope = result
            col.append(new_col)
            return new_result, new_row, col, row_scope

        elif isinstance(result, tuple) and len(result) == 3:
            new_result, new_row, new_col = result
            col.append(new_col)
            return new_result, new_row, col, row_scope
        elif isinstance(result, tuple) and len(result) == 2:
            new_result, new_col = result
            col.append(new_col)
            return new_result, row, col, row_scope
        else:
            return result, row, col, row_scope

    def to_str(self):
        arg_list = []
        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    arg_list.append('all_rows')
                else:
                    arg_list.append(each_child[1])
            else:
                sub_result = each_child[1].to_str()
                # print ("exit func: ", each_child[1].func)

                arg_list.append(sub_result)

        result = APIs[self.func]["tostr"](*arg_list)
        return result

    def _mutate_dict(self, dict_in, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2):
        new_dict = {}
        # mutate function
        new_func = dict_in['func']
        if random.random() > alpha:
            for arg, ops in self.swap_dict.items():
                if dict_in['func'] in ops:
                    swap_func = random.choice(ops)  # have chance not changing
                    new_func = swap_func
                    break
        new_dict['func'] = new_func

        # deal with args
        new_dict['args'] = []
        for each_child in dict_in["args"]:
            if isinstance(each_child, str):
                new_child = each_child
                # mutate int
                if each_child.isnumeric() and random.random() < theta:
                    new_child = max(int(each_child) + random.randint(-10, 10), 0)
                    new_child = str(new_child)  # TODO: float numbers

                # mutate columns
                cols = self.full_table.columns
                if each_child in cols:
                    if random.random() > beta:
                        new_child = random.choice(cols)  # have chance not changing
                        # TODO: content mutation
                new_dict['args'].append(new_child)

            elif isinstance(each_child, dict):
                new_child = self._mutate_dict(each_child)
                new_dict['args'].append(new_child)
            else:
                raise ValueError("child type error")

        return new_dict

    def mutate(self, mutate_num_max=500, alpha=0.5, beta=0.5, gamma=0.6, theta=0.15, omega=0.2):
        mutations = []
        visited_node = set()
        for i in range(mutate_num_max):
            new_dict = self._mutate_dict(self.dict_in, alpha=alpha, beta=beta, gamma=gamma, theta=theta, omega=omega)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval())
                except:
                    continue
                # print(new_result)
                if 'ExeError():' not in new_result:
                # if new_result == 'True':
                    mutations.append(new_node)
                    break
        return mutations

    def _str_replace(self, dict_in, perm):
        new_dict = {}
        # mutate function
        new_func = dict_in['func']
        if dict_in['func'] in func_map_str_replace:
            if perm[0] == 1:
                swap_func =  func_map_str_replace[new_func]
                new_func = swap_func
                perm = perm[1:]
            else:
                perm = perm[1:]
        new_dict['func'] = new_func

        # deal with args
        new_dict['args'] = []
        for each_child in dict_in["args"]:
            if isinstance(each_child, str):
                new_child = each_child
                new_dict['args'].append(new_child)
            elif isinstance(each_child, dict):
                new_child = self._str_replace(each_child, perm)
                new_dict['args'].append(new_child)
            else:
                raise ValueError("child type error")

        return new_dict

    def _num_to_replace(self, dict_in):
        '''

        :param dict_in:
        :return: count of alternative function names that can be replaced
        '''
        count = 0
        func = dict_in['func']
        if func in func_map_str_replace:
            count += 1
        for each_child in dict_in["args"]:
            if isinstance(each_child, dict):
                count += self._num_to_replace(each_child)
        return count

    def str_rep(self):
        '''
        Try to replace functions like "eq" into "str_eq"
        Returns:

        '''
        mutations = set()
        visited_node = set()
        num_to_replace = self._num_to_replace(self.dict_in)
        #Generate full permutations of n-digit binary numbers
        #Each elem in a perm indicates whether (1) or not (0) to replace an alternative func
        s = list(it.product(range(2), repeat=num_to_replace))
        for perm in s:
            new_dict = self._str_replace(self.dict_in, perm)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval_index()[0])
                except:
                    continue
                # print(new_result)
                # if 'ExeError():' not in new_result:
                mutations.add(new_node)
                if new_result == 'True':
                    # print("one correct")
                    return True

        return False

    def str_rep_form(self):
        mutations = set()
        visited_node = set()
        num_to_replace = self._num_to_replace(self.dict_in)
        #Generate full permutations of n-digit binary numbers
        #Each elem in a perm indicates whether (1) or not (0) to replace an alternative func
        s = list(it.product(range(2), repeat=num_to_replace))
        for perm in s:
            new_dict = self._str_replace(self.dict_in, perm)
            if str(new_dict) not in visited_node:
                visited_node.add(str(new_dict))
                new_node = Node(self.full_table, new_dict)
                # test node
                try:
                    new_result = str(new_node.eval_index()[0])
                except:
                    continue
                # print(new_result)
                # if 'ExeError():' not in new_result:
                mutations.add(new_node)
                if new_result == 'True':
                    # print("one correct")
                    return new_dict
        return False

def to_str_all(json_in):
    '''
	transform all logic forms into strings
	'''

    with open(json_in) as f:
        data_in = json.load(f)

    num_all = 0
    num_correct = 0

    for data in tqdm(data_in):

        num_all += 1
        logic = data["logic"]
        logic_str = data['logic_str']

        table_header = data["table_header"]
        table_cont = data["table_cont"]

        try:
            pd_in = defaultdict(list)
            for ind, header in enumerate(table_header):
                for inr, row in enumerate(table_cont):

                    # remove last summarization row
                    if inr == len(table_cont) - 1 \
                            and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or \
                                 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                        continue
                    pd_in[header].append(row[ind])

            pd_table = pd.DataFrame(pd_in)
        except Exception:
            continue

        root = Node(pd_table, logic)
        res = root.to_str()

        if res == logic_str[:-7]:
            num_correct += 1
        else:
            print(res)
            print(logic_str)

    print("All: ", num_all)
    print("Correct: ", num_correct)

    print("Correctness Rate: ", float(num_correct) / num_all)

    return num_all, num_correct



def execute(data):
    '''
    Execute a logical form on a table
    Args:
        data:

    Returns:

    '''
    logic = data["logic"]
    table_header = data["table_header"]
    table_cont = data["table_cont"]

    try:
        pd_in = defaultdict(list)
        for ind, header in enumerate(table_header):
            for inr, row in enumerate(table_cont):
                # remove last summarization row
                if inr == len(table_cont) - 1 \
                        and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
                             "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                    continue
                pd_in[header].append(row[ind])

        pd_table = pd.DataFrame(pd_in)
    except Exception:
        return False


    root = Node(pd_table, logic)
    res = root.eval_index()
    res = res[0]

    if 'ExeError' in str(res) or not res:
        # The initial trial of execution is based on raw function names like "eq", "filter_eq"
        # However, sometimes "str_eq" or "filter_str_eq" should be the real function name
        # str_eq() enables enumerating all possible replacements such as "eq" to "str_eq"
        res = root.str_rep()
    return res



def tostr(data):
    logic = data["logic"]

    table_header = data["table_header"]
    table_cont = data["table_cont"]

    try:
        pd_in = defaultdict(list)
        for ind, header in enumerate(table_header):
            for inr, row in enumerate(table_cont):

                # remove last summarization row
                if inr == len(table_cont) - 1 \
                        and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
                             "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                    continue
                pd_in[header].append(row[ind])

        pd_table = pd.DataFrame(pd_in)
    except Exception:
        return False

    root = Node(pd_table, logic)
    res = root.to_str()
    return res




import os, io, re, subprocess
import json
import logging
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from pyrouge import Rouge155
import pandas as pd


def bleu_score(labels_file, predictions_path):
    bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    try:
        with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
            bleu_out = subprocess.check_output(
                [bleu_script, labels_file],
                stdin=predictions_file,
                stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            return float(bleu_score)

    except subprocess.CalledProcessError as error:
        return None


def pattern_match(logic, truth):

    def extract_pattern(logic):
        logic = logic.strip()[:-7]
        API_words = [_ for _ in APIs.keys()]
        API_words += ['hop']
        key_words = ['{', '}', 'all_rows', ';']
        temp = []
        for i, x in enumerate(logic.split()):
            if x in key_words or (x in API_words and i < len(logic.split()) - 1 and logic.split()[i + 1] == "{"):
                temp.append(x)
        return temp

    logic = extract_pattern(logic)
    truth = extract_pattern(truth)
    if len(logic) != len(truth):
        return False
    for a, b in zip(logic, truth):
        if a != b:
            return False
    return True


def validation_task(val_file, model, tokenizer, split, args):
    if args.task == 'summ':
        val_dataset = ScigenDataset(val_file, tokenizer, args.max_src_len, args.max_tgt_len, args.add_type, args.pre_com)
    else: 
        val_dataset = ContlogDataset(val_file, tokenizer, args.max_src_len, args.max_tgt_len, args.task, args.add_type, args.pre_com)
    val_loader = DataLoader(val_dataset,
                            num_workers=5,
                            batch_size=args.batch_size,
                            shuffle=False)
    model.eval()
    pred_list = []
    ref_list = []

    # create files for scripts
    gt = open(os.path.join(args.log_path, args.affix, f'references_{split}.txt'), 'w')
    pred = open(os.path.join(args.log_path, args.affix, f'predictions_{split}.txt'), 'w')
    pred_split = os.path.join(args.log_path, args.affix, f'predictions_{split}/')
    ref_split = os.path.join(args.log_path, args.affix, f'references_{split}/')
    if not os.path.exists(pred_split):
        os.makedirs(pred_split)
    if not os.path.exists(ref_split):
        os.makedirs(ref_split)
    k = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            y = batch['target_ids'].to(args.device, dtype=torch.long)
            ids = batch['source_ids'].to(args.device, dtype=torch.long)
            mask = batch['source_mask'].to(args.device, dtype=torch.long)
            samples = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=args.max_tgt_len,
                num_beams=4,
                early_stopping=False
            )


            for reference, s in zip(y, samples):
                with open(ref_split + str(k) + '_reference.txt', 'w') as sr, \
                        open(pred_split + str(k) + '_prediction.txt', 'w') as sw:
                    reference = tokenizer.decode(reference, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                    gt.write(reference.lower() + '\n')
                    sr.write(reference.lower() + '\n')
                    ref_list.append(reference.lower())
                    text = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_list.append(text.lower())
                    pred.write(text.lower() + '\n')
                    sw.write(text.lower() + '\n')
                    k += 1
        gt.close()
        pred.close()

        if args.task == 'text' or args.task == 'summ':
            gt = os.path.join(args.log_path, args.affix, f'references_{split}.txt')
            pred = os.path.join(args.log_path, args.affix, f'predictions_{split}.txt')

            # ROUGE scripts
            r = Rouge155()
            r.system_dir = os.path.join(args.log_path, args.affix, f'predictions_{split}/')
            r.model_dir = os.path.join(args.log_path, args.affix, f'references_{split}/')
            # define the patterns
            r.system_filename_pattern = '(\d+)_prediction.txt'
            r.model_filename_pattern = '#ID#_reference.txt'
            logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
            results_dict = r.convert_and_evaluate()
            rouge_result = "\n".join(
                [results_dict.split("\n")[3], results_dict.split("\n")[7], results_dict.split("\n")[15],
                 results_dict.split("\n")[19]])
            print("[INFO] Rouge scores: \n", rouge_result)
            # log_file.write(rouge_result + '\n')
            results_dict = results_dict.split("\n")
            rouge_score_list = []

            for i in [3, 7, 15, 19]:
                results = results_dict[i]
                rouge_score = float(results.split()[3])
                rouge_score_list.append(rouge_score * 100)

        # If the task is table-to-logic generation
        elif args.task == 'logic':
            num_samples = 0  # all samples
            num_p_correct = 0  # pattern-match examples
            num_exact = 0  # exact-match examples
            num_e_correct = 0  # execution-correct examples
            with open(val_file, 'r') as fp:
                data = json.load(fp)
                for d, logic_str in zip(data, pred_list):
                    num_samples += 1
                    # exact match
                    if logic_str == d['logic_str']:
                        num_exact += 1
                        num_p_correct += 1
                        num_e_correct += 1
                    else:
                        # pattern match
                        label = pattern_match(logic_str, d['logic_str'])
                        if label:
                            num_p_correct += 1
                        try:
                            # execution accuracy evaluate
                            cur_logic = parse_str(logic_str[:-7], func_map)
                            cur_execute_batch = {"table_header": d['table_header'],
                                                 "table_cont": d['table_cont'],
                                                 "logic": cur_logic}
                            res = execute(cur_execute_batch)
                            if res == True:
                                num_e_correct += 1
                        except:
                            continue
                acc_e = 1. * num_e_correct / num_samples * 100
                acc_p = 1. * num_p_correct / num_samples * 100
                acc_exact = 1. * num_exact / num_samples * 100
            print("[INFO] Execution accuracy:  ", acc_e)
            print("[INFO] Pattern accuracy: ", acc_p)
            print("[INFO] Exact-match accuracy:  ", acc_exact)


        val_metric_dict = {}
        if args.task == 'text' or args.task == 'summ':
            for type, score in zip(['1', '2', '4', 'L'], rouge_score_list):
                val_metric_dict[f'rouge{type}'] = score
            # val_metric_dict['bleu4'] = bleu4
        elif args.task == 'logic':
            val_metric_dict = {"exec_acc": acc_exact,
                               "pat_acc": acc_p,
                               "LF_acc": acc_e
                               }
        else:
            raise NotImplementedError
        model.train()
        return val_metric_dict

##############################################################################
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def segment_text(text):
    sentences = sent_tokenize(text)
    return [sentence.strip() for sentence in sentences]

# rule based logical type classifier
def classify_logical_type(text):
    logic_patterns = {
        'majority': r'\b(majority|most|predominantly|greater part)\b',
        'superlative': r'\b(among|max|highest|largest|biggest|greatest|latest|minimum|min|lowest|smallest|least|fewest|greater)\b',
        'comparative': r'\b(comparison|compared|than|better|worst)\b',
        'aggregation': r'\b(sum|total|aggregate|combined|summed|overall)\b',
        'ordinal': r'\b(ordinal|rank|ranking|order|position|first|second|third)\b',
        'count': r'\b(count|number|total|frequency|amount)\b',
        'unique': r'\b(unique|distinct|different|only|single)\b',
    }

    for logic_type, pattern in logic_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return logic_type

    return 'unique' 

import re

# Highlight matching table cells based on word-level token comparison
def highlight_tabular_data(segmented_data, table_contents_value):
    highlight_cells = []

    for row_index, row in enumerate(table_contents_value):
        for col_index, cell_value in enumerate(row):
            cell_value_cleaned = str(cell_value).strip()  # Clean the cell value for comparison
            
            # Tokenize the cell value for comparison
            cell_value_tokens = set(cell_value_cleaned.split())  # Use a set for quick lookup

            for segment in segmented_data:
                segment_text = segment['text']
                # Tokenize the segment text
                segment_tokens = set(re.findall(r'\b\w+\b', segment_text))  # Extract words as tokens

                # Check for intersection between cell tokens and segment tokens
                if cell_value_tokens.intersection(segment_tokens):
                    # If there's a match, highlight all cells in the current row
                    for col in range(len(row)):
                        highlight_cells.append({
                            'row': row_index,
                            'column': col,
                            'text': str(row[col]).strip(),  # Highlight all cells in this row
                            # Optional: include additional information if needed
                            # 'max_rank': 3,
                            # 'min_rank': -5
                        })
                    break  # No need to check other segments for this row, move to the next row

    return highlight_cells

import json
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

def linearize_table_data(data: dict, add_type: bool = False, pre_com: bool = False) -> dict:
    # Extract the json structure
    table_caption = data["table_caption"]
    table_header = data["table_column_names"]
    table_contents = data["table_content_values"]
    textual_data = data["long_text"]

    # Initialize source text with the caption
    src_text = f"<table> <caption> {str(table_caption)} </caption> "

    # # convert table data to pandas for easier indexing
    # pd_in = defaultdict(list)
    # for ind, header in enumerate(table_header):
    #     for row in table_contents:
    #         pd_in[header].append(row[ind])
    #     else:
    #         pd_in[header].append(None)
    # pd_table = pd.DataFrame(pd_in)
    #
    # # Linearize the entire table (exclude highlights for now)
    # for row_idx in range(len(pd_table)):
    #     for col_header in pd_table.columns:
    #         cell_value = pd_table[col_header].iloc[row_idx]
    #         # Construct cell string for each cell in the table
    #         cell_str = f"<cell> {cell_value} <col_header> {col_header} </col_header> <row_idx> {row_idx} </row_idx> </cell> "
    #         src_text += cell_str

    # Initialize a list to hold the linearized data
    linearized_data = []

    # Determine the number of rows (assuming all rows are of equal length)
    num_rows = len(table_contents)

    # Create a linearized representation of the table
    for row_idx in range(num_rows):
        for header in table_header:
            # Get the index of the current header
            header_index = table_header.index(header)
            # Get the corresponding cell value, or None if out of bounds
            cell_value = table_contents[row_idx][header_index] if header_index < len(table_contents[row_idx]) else None
            
            # Construct cell string for each cell in the table
            cell_str = f"<cell> {cell_value} <col_header> {header} </col_header> <row_idx> {row_idx} </row_idx> </cell> "
            linearized_data.append(cell_str)

    # Join all cell strings together and add to source text
    src_text += ''.join(linearized_data)

    # Add caption at the end of the table linearization
    src_text += f"</table> {textual_data}"

    # Assign linearized text to 'src_text' field in data
    data['src_text'] = src_text

    return data


def old_linearize_table_data(data: dict, add_type: bool = False, pre_com: bool = False) -> dict:
    '''
    Args:
        data: input JSON structure containing the table and highlighted data.
        add_type: whether to add logic type information.
        pre_com: whether to add pre-computed information (max_rank, min_rank).
    
    Returns:
        The same data dictionary with the linearized table and text added to 'src_text'.
    '''
    # Extract the table structure
    table_caption = data["table_caption"]
    table_header = data["table_column_names"]
    table_contents = data["table_content_values"]
    highlight_cells = data['highlight_cells']
    
    # Initialize source text with the caption
    src_text = f"<table> <caption> {table_caption} </caption> "
    
    # If logical type information is needed
    if add_type:
        for seg in data['segmented_text']:
            src_text += f"<type> {seg['action']} </type> {seg['text']} "

    # Convert table data to pandas for easier indexing
    pd_in = defaultdict(list)
    for ind, header in enumerate(table_header):
        for row in table_contents:
            pd_in[header].append(row[ind])
    #pd_table = pd.DataFrame(pd_in)

    # Process highlighted cells
    for cell in highlight_cells:
        row = cell['row']
        col_idx = cell['column']
        cell_value = cell['text']
        #max_rank = cell.get('max_rank', None)
        #min_rank = cell.get('min_rank', None)
        
        col_header = table_header[col_idx]  # Column header from table structure
        
        # Construct cell string
        #if pre_com and max_rank is not None:
        #    cell_str = f"<cell> {cell_value} <col_header> {col_header} </col_header> <row_idx> {row} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank></cell> "
        #else:
        cell_str = f"<cell> {cell_value} <col_header> {col_header} </col_header> <row_idx> {row} </row_idx> </cell> "
        
        # Append cell data to source text
        src_text += cell_str
    
    # Close the table tag
    src_text += "</table>"
    
    # Add the linearized output to the 'src_text' field in the data dictionary
    data['src_text'] = src_text
    
    return data


def old_preprocess_summarization(data_file: str, add_type: bool, pre_com: bool):
    with open(data_file, 'r') as f:
        data = json.load(f)

        preprocessed_data = []

        # Process each item in the dataset
        for key, item in tqdm(data.items()):
            # Extract relevant fields
            text = item['text']
            table_caption = item['table_caption']
            table_column_names = item['table_column_names']
            table_content_values = item['table_content_values']
            
            # Segment the text
            segmented_data = segment_text(text)
            
            # Classify logical types for segmented sentences
            segmented_data_with_actions = [
                {
                    'text': sentence,
                    'action': classify_logical_type(sentence)
                } for sentence in segmented_data
            ]
            
            # Highlight table data based on segmented sentences
            highlight_cells = highlight_tabular_data(segmented_data_with_actions, table_content_values)

            # Prepare the data structure
            processed_item = {
                'text': text,
                'table_caption': table_caption,
                'table_column_names': table_column_names,
                'table_content_values': table_content_values,
                'segmented_text': segmented_data_with_actions,
                'highlight_cells': highlight_cells
            }

            # Linearize the table and add src_text
            processed_item = linearize_table_data(processed_item, add_type, pre_com)

            # Add the processed item to the preprocessed data list
            preprocessed_data.append(processed_item)

        return preprocessed_data

def preprocess_summarization(data_file: str, add_type: bool, pre_com: bool):
    with open(data_file, 'r') as f:
        data = json.load(f)

        preprocessed_data = []

        # Process each item in the dataset
        for key, item in tqdm(data.items()):
            # Extract relevant fields
            table_caption = item['table_caption']
            table_column_names = item['table_column_names']
            table_content_values = item['table_content_values']
            textual_data = item['long_text']
            text = item['text']
            
            # Segment the text
            segmented_data = segment_text(text)
            
            # Classify logical types for segmented sentences
            segmented_data_with_actions = [
                {
                    'text': sentence,
                    'action': classify_logical_type(sentence)
                } for sentence in segmented_data
            ]
            
            # Highlight table data based on segmented sentences
            highlight_cells = highlight_tabular_data(segmented_data_with_actions, table_content_values)

            # Prepare the data structure
            processed_item = {
                'table_caption': table_caption,
                'table_column_names': table_column_names,
                'table_content_values': table_content_values,
                'long_text': textual_data,
                'text': text,
            }

            # Linearize the table and add src_text
            processed_item = linearize_table_data(processed_item, add_type, pre_com)

            # Add the processed item to the preprocessed data list
            preprocessed_data.append(processed_item)

        return preprocessed_data

class ScigenDataset(Dataset):
    def __init__(self, file, tokenizer, max_src_len, max_tgt_len, add_type, pre_com):
        self.data = preprocess_summarization(file, add_type, pre_com)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]

        # Get the source text from the linearized data
        src_text = d['src_text'].strip()
        src_text = ' '.join(src_text.split())

        tgt_text = d['text'].strip()
        tgt_text = ' '.join(tgt_text.split())

        # Tokenize source and target texts
        source = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_len
        )
        target = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tgt_len
        )

        # Convert tokenized data to tensors
        source_input_ids = torch.LongTensor(source.data["input_ids"])
        target_input_ids = torch.LongTensor(target.data["input_ids"])
        source_mask = torch.LongTensor(source.data["attention_mask"])

        return {
            'source_ids': source_input_ids,
            'source_mask': source_mask,
            'target_ids': target_input_ids
        }

##############################################################################

import math
import random
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Adafactor, AdamW

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='facebook/bart-base', type=str, help="specify the pretrained model to use")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization and reproducibility")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to perform training")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to perform validation")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to perform testing")
    parser.add_argument('--pretrain', default=False, action="store_true", help="whether to train or test the model")

    parser.add_argument('--optimizer', default='Adamw', choices=['Adamw', 'Adafactor'], type=str)
    parser.add_argument('--epoch', default=1, type=int, help="number of epochs for training")
    parser.add_argument('--batch_size', default=5 , type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--every', default=50, type=int, help="interval for evaluation")
    parser.add_argument('--interval_type', default='step', type=str, choices=['step', 'epoch'], help="whether to evaluate at intervals based on steps or epochs")
    parser.add_argument('--interval_step', default=1000, type=int, help="interval for evaluation when interval_type = step.")
    parser.add_argument('--load_from', default=None, type=str, help="model checkpoint path")
    parser.add_argument('--max_src_len', default=1024, type=int, help="max length of input sequence")
    parser.add_argument('--max_tgt_len', default=200, type=int, help="target output length")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="number of steps to accumulate gradients before updating parameters")

    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/Dataset/contlog')
    parser.add_argument('--log_path',type=str, default='/content/drive/MyDrive/Output/logs/d2t/outputs')
    parser.add_argument('--ckpt_path', type=str, default='/content/drive/MyDrive/Output/models/d2t')
    parser.add_argument('--affix', type=str, default=None, required=True, help="The experiment name")
    parser.add_argument('--device', type=str, default='cuda', help="specifies the device to use for computations (CUDA only)")
    parser.add_argument('--n_gpu', type=str, default=0, help="number of GPU to use")
    parser.add_argument('--task', type=str, default='text', help='task: text (table2text) or logic (table2logic) or summ (tabletextsumm)')
    parser.add_argument('--add_type', default=False, action="store_true", help="indicate whether to add type information to the input")
    parser.add_argument('--pre_com', default=False, action="store_true", help="whether to do numerical precomputation")
    parser.add_argument('--global_step', default=1, type=int, help="initialize global step counter")
    parser.add_argument('--use_cache', default=False, action="store_true", help="enable caching mechanisms")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    set_seed(args) # for reproducibility

    # add special tokens to tokenizer for linearized table structure
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    markers = ["{", "}", "<table>", "</table>", "<type>", "</type>", "<cell>", "</cell>", "<col_header>", "</col_header>", "<row_idx>", "</row_idx>"]
    if args.pre_com:
        markers += ["<max_rank>", "</max_rank>", "<min_rank>", "</min_rank>", "<sum_cell>", "</sum_cell>", "<avg_cell>", "</avg_cell>"]
    tokenizer.add_tokens(markers)

    # model setup, load pretrained weights, and accomodate special tokens
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    if args.load_from is not None:
        model.load_state_dict(torch.load(args.load_from))

    # layer freezing for retaining features
    def freeze_params(model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False

    # loss function
    #criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    # create directories to store logs and models
    if not os.path.exists(os.path.join(args.log_path, args.affix)):
        os.makedirs(os.path.join(args.log_path, args.affix))
    if not os.path.exists(os.path.join(args.ckpt_path, args.affix)):
        os.makedirs(os.path.join(args.ckpt_path, args.affix))

    # set pretraining data location
    if args.task == 'logic':
        train_file = os.path.join(args.data_path, 'all_pretrain_train_s.json')
        val_file = os.path.join(args.data_path, 'all_pretrain_valid.json')
        test_file = os.path.join(args.data_path, 'all_pretrain_test.json')
    elif args.task == 'text':
        train_file = os.path.join(args.data_path, 'train.json')
        val_file = os.path.join(args.data_path, 'val.json')
        test_file = os.path.join(args.data_path, 'test.json')
    elif args.task == 'summ':
        train_file = os.path.join(args.data_path, 'train-400.json')
        val_file = os.path.join(args.data_path, 'valid-50.json')
        test_file = os.path.join(args.data_path, 'test-50.json')
    else:
        raise NotImplementedError

    if args.do_train:
        # freeze embedding layers
        if args.model.startswith('t5'):
            for d in [model.encoder]:
                freeze_params(d.embed_tokens)
        else:
            for d in [model.model.encoder, model.model.decoder]:
                # freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

        if args.task == 'summ':
            train_dataset = ScigenDataset(train_file, tokenizer, args.max_src_len, args.max_tgt_len, args.add_type, args.pre_com)
        else: 
            train_dataset = ContlogDataset(train_file, tokenizer, args.max_src_len, args.max_tgt_len, args.task, args.add_type, args.pre_com)
        train_loader = DataLoader(train_dataset, num_workers=5, batch_size=args.batch_size, shuffle=True)
        model.train()

        if args.optimizer == 'Adamw':
            optimizer = AdamW(model.parameters(), args.learning_rate)
        elif args.optimizer == 'Adafactor':
            optimizer = Adafactor(model.parameters(), args.learning_rate, relative_step=False)
        else:
            raise NotImplementedError

        global_step = 0
        total_loss = []
        # best validation score
        best_val = 0
        # best_metric = "bleu4" if args.task == 'text' else "exec_acc"

        for epoch_idx in range(1, args.epoch+1):
            print("[INFO] start training {}th epoch".format(epoch_idx))
            for idx, batch in enumerate(tqdm(train_loader)):
                # data preparation
                lm_labels = batch['target_ids'].to(args.device, dtype=torch.long)
                lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                ids = batch['source_ids'].to(args.device, dtype=torch.long)
                mask = batch['source_mask'].to(args.device, dtype=torch.long)

                # forward pass and loss calculation
                outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                # gradient accumulation, loss scaling, and backpropagation
                loss = loss / args.gradient_accumulation_steps
                total_loss.append(loss.item())
                loss.backward()

                # optimizer update model parameter using gradients every gradient accumulation step 
                if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()
                
                # Perplexity - measure of how well the model is able to predict a sequence every args.every
                if idx % args.every == 0 and idx > 0:
                    perplexity = math.exp(np.mean(total_loss))
                    total_loss = []

                # save model every args.interval_type step or epoch
                if (args.interval_type == 'step' and global_step % args.interval_step == 0 and global_step > 0) \
                    or (args.interval_type == 'epoch' and (idx + 1) == len(train_loader)):
                    if args.interval_type == 'step':
                        torch.save(model.state_dict(), '{}/{}/{}_step{}.pt'.format(args.ckpt_path, args.affix, args.model.split('/')[-1], global_step))
                    else:
                        torch.save(model.state_dict(), '{}/{}/{}_ep{}.pt'.format(args.ckpt_path, args.affix, args.model.split('/')[-1], epoch_idx))

                    # validation step
                    val_scores = validation_task(val_file, model, tokenizer, 'valid', args)
                    if args.task == "logic" and val_scores["exec_acc"] > best_val:
                        best_val = val_scores["exec_acc"]
                    test_scores = validation_task(test_file, model, tokenizer, 'test', args)
                global_step += 1

    # just test model, no training
    if args.do_test:
        test_scores = validation_task(test_file, model, tokenizer, 'test', args)
        print(test_scores)
