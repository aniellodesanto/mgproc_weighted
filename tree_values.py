#!/usr/bin/env python
# -*- coding: utf-8 -*-

# metrics as originally defined
# This file is called by metrics.py
#
# It defines a general function memory_measure from which various the values
# for various memory-based metrics can be computed based on how its arguments
# are instantiated.
#
# Metric   | Operator        | Memory Type
# ---------|-----------------|----------------
# MaxT     | safemax         | tenure_extract
# SumT     | sum             | tenure_extract
# BoxT     | len             | tenure_extract
# AvgT     | avg             | tenure_extract
# MaxTR    | None/sorted     | tenure_extract
# --------------------------------------------
# MaxS     | safemax         | move_extract
# SumS     | sum             | move_extract
# Movers   | len             | move_extract
# AvgS     | avg             | move_extract
# MaxSR    | None/sorted     | move_extract
#
# While all the functions in this module are meant to be private,
# they are not prefixed with _ so that the user can easily reference them
# in text files to define various metrics.

from io_tree import IONode, IOTree


####################
#  Math Operators  #
####################

def safemax(*args: 'list of ints') -> int:
    """max that returns 0 if there are no arguments"""
    try:
        return max(*args)
    except:
        return 0


def safediv(dividend: int, divisor: int) -> float:
    """safe division by 0"""
    if divisor != 0:
        return dividend / divisor
    else:
        return 0


def avg(int_list: list) -> float:
    """Compute the average of a list of integers."""
    return safediv(sum(int_list), len(int_list))


########################
#  Matching Functions  #
########################

def typedict(IONode) -> dict:
    """Compute dictionary of Bools encoding type of IONode.

    Possible node types are
    I: interior (= not a leaf node)
    U: unpronounced leaf (= empty head)
    P: pronounced leaf
    F: part of functional projection (e.g. T')
    C: part of content projection (e.g. V')

    The returned dictionary has exactly one of the three set to True.

    Examples
    --------
    >>> node = IONode(empty=True, leaf=True)
    >>> typedict(node)
    {'I': False, 'P': False, 'U': True}
    >>> node.leaf = False
    >>> typedict(node)
    {'I': True, 'P': False, 'U': False}
    >>> node.leaf = True
    >>> node.empty = False
    >>> typedict(node)
    {'I': False, 'P': True, 'U': False}
    """
    types = {'I': False, 'U': False, 'P': False, 'F': False, 'C': False}
    if IONode.leaf == False:
        types['I'] = True
    elif IONode.empty == True:
        types['U'] = True
    else:
        types['P'] = True

    if IONode.content == True:
        types['C'] = True
    else:
        types['F'] = True

    return types


def matches_types(IONode, node_types: list=None) -> bool:
    """Check whether IOnode matches at least one of the listed node types.

    The only standardized node types are
    I: interior (= not a leaf node)
    U: unpronounced leaf (= empty head)
    P: pronounced leaf
    F: part of functional projection
    C: part of content projection

    If node_types contains any other entries, they are treated as if
    their value were False.

    Examples
    --------
    >>> node = IONode(empty=True, leaf=True)
    >>> matches_types['I','U']
    True
    >>> matches_types['I','P']
    False
    >>> node.leaf = False
    >>> matches_types['I','P']
    True
    >>> matches_types['U']
    False
    """
    if node_types:
        return max([typedict(IONode).get(node_type, False)
                    for node_type in node_types])
    else:
        return False


##########################
#  Raw Value Extraction  #
##########################

def tenure_extract(IOTree, filters: list=[], trivial: bool=False) -> dict:
    """Compute dict of tenure values for all nodes in IOTree.

    The dictionary is of the form address: tenure.

    Parameters
    ----------
    IOTree : IOTree
        index/outdex annotated Gorn tree for which values are to be computed
    filters : list of str
        do not consider the values of nodes whose type is listed here
    trivial : bool
        whether to include nodes with trivial tenure (= tenure 1 or 2)

    Examples
    --------
    >>> tree = tree_from_file('./examples/ugly')
    >>> tenure_extract(tree)
    {'121': 12, '122111': 11, '1221223': 3, '122122': 3, '1222': 24, '11': 10,
     '1221122': 3, '122113': 13, '12212': 16}
    >>> tenure_extract(tree, trivial=True)
    {'': 1, '121': 12, '11': 10, '1211': 1, '1221122': 3, '122': 1,
     '122113': 13, '1221111': 1, '1221221': 1, '12211221': 1, '122111': 11,
     '1221': 1, '1221222': 2, '112': 2, '1': 1, '12211': 1, '12': 1, '111': 1,
     '12211211': 1, '1222': 24, '1221223': 3, '122112': 1, '1221211': 1,
     '1221121': 1, '12212': 16, '12221': 1, '1221131': 1, '122121': 1,
     '122122': 3}
    >>> tenure_extract(tree, filters=['P'], trivial=True)
    {'': 1, '121': 12, '11': 10, '122112': 1, '1221122': 3, '122': 1,
     '122113': 13, '1221121': 1, '12212': 16, '122111': 11, '1221': 1,
     '122121': 1, '1': 1, '12211': 1, '122122': 3, '12': 1, '1222': 24}
    >>> tenure_extract(tree, filters=['I', 'P'], trivial=True)
    {}
    """
    threshold = 2 if not trivial else 0
    return {node.address: node.tenure() for node in IOTree.struct.values()
            if not matches_types(node, filters) and node.tenure() > threshold}


def move_length(IOTree, IONode, filters: list=[], trivial: bool=False) -> dict:
    """Compute dict of size values for given IONode in IOTree.

    The dictionary is of the form triple: value, where triple has the form 
    (address of mover, address of target, movement feature). Note that the
    dictionary is flat rather than nested to make it behave just like the
    dictionary produced by tenure_extract --- the unwieldy keys are the price
    we pay for this design.

    Parameters
    ----------
    IOTree : IOTree
        index/outdex annotated Gorn tree within which values are to be computed
    IONode: IONode
        node in IOTree whose move length is to be evaluated
    filters: list of str
        do not consider move steps that were triggered by one of these features
    trivial : bool
        whether to consider intermediate movement steps

    Examples
    --------
    >>> tree = tree_from_file('./examples/ugly')
    >>> node = tree.struct['122112']
    >>> move_length(tree, node)
    {('122112', '', 'top'): 6}
    >>> move_length(tree, node, trivial=True)
    {('122112', '122', 'acc'): 3, ('122112', '', 'top'): 6
    >>> move_length(tree, node, filters=['nom', 'top'], trivial=True)
    {('122112', '122', 'acc'): 3}
    >>> move_length(tree, node, filters=['nom', 'acc', 'top'], trivial=True)
    {}
    """
    # only keep non-final movers if trivial is set to True
    try:
        final_target = [next(reversed(IONode.movement))]
    except:
        final_target = []
    move_targets = IONode.movement if trivial else final_target

    return {(IONode.address, target, IONode.movement[target]):
            abs(IONode.index() - IOTree.struct[target].index())
            for target in move_targets
            if IONode.movement[target] not in filters}


def move_extract(IOTree, filters: list=[], trivial: bool=False) -> dict:
    """Compute dict of size values for all nodes in IOTree.

    See move_length for details about the format of the dictionary.

    Parameters
    ----------
    IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
    filters: list of str
        do not consider move steps that were triggered by one of these features
    trivial : bool
        whether to consider intermediate movement steps

    Examples
    --------
    >>> tree = tree_from_file('./examples/right_embedding')
    >>> move_extract(tree)
    {('212112121211212121', '212112121211212', 'nom'): 3,
     ('212112121211', '21211212', 'nom'): 4, ('21211', '2', 'nom'): 5,
     ('21211212121121', '2121121212', 'extra'): 13,
     ('21211212121121212222', '2121121212112', 'hn'): 15,
     ('2121121', '212', 'extra'): 5, ('21211212121222', '212112', 'hn'): 9}
    >>> move_extract(tree, filters=['nom', 'hn'])
    {('21211212121121', '2121121212', 'extra'): 13,
     ('2121121', '212', 'extra'): 5}
    >>> move_extract(tree, filters=['nom', 'hn', 'extra'], trivial=True)
    {}
    >>>tree = tree_from_file('./trees/examples/shortNOI')
    move_extract(tree, trivial=True)
    {('12121', '12', 'nom'): 3, ('12121', '', 'wh'): 5, ('12122222121', '12122222', 'nom'): 4}
    """
    movers = {}
    for node in IOTree.struct.values():
        new = move_length(IOTree, node, filters=filters, trivial=trivial)
        movers.update(new)
    return movers

######################################
# REACTIVATION FUNCTIONS: CHONG STYLE #
#######################################

def node_reactivation_plain(IOTree,IONode, features_movers, trivial: bool=False) -> dict:
    """Compute dict of reactivation values for given IONode in IOTree.
        Reactivation(m) = i(m) - o(n). Default Reactivation is set to null, following Zhang (2017).
        
        The dictionary is of the form triple: value, where triple has the form
        (address of mover, address of previous mover, movement feature).
        
        Note that most movers will have just one value, but the dictionary is structured
        this way for consistency with those extracted by move_length.
        
        """
    #node_features = IONode.movement
    # only keep non-final movers if trivial is set to True
    try:
        final_target = [next(reversed(IONode.movement))]
    except:
        final_target = []
    node_features = IONode.movement if trivial else final_target

    reactivation = {}
    for target in node_features: #node_features.keys()
        feature = IONode.movement[target]
        #isolate list of movers per specific feature
        feature_movers = features_movers[feature]
        previous = ''
        if feature_movers.index(IONode.address) != 0:
            previous = feature_movers[feature_movers.index(IONode.address)-1]
            react_value =  abs(IONode.index() - IOTree.struct[previous].outdex())
            reactivation.update({(IONode.address,previous,feature):react_value})

    return reactivation

def reactivation_plain_extract(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of reactivation values for all nodes/all features in  IOTree.
        Reactivation(m) = i(m) - o(n). Default Reactivation is set to null, following Zhang (2017).
        See node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>>tree = tree_from_file('./trees/examples/shortNOI')
        
        >>> reactivation_plain_extract(tree)
        {}
        
        >>> reactivation_plain_extract(tree, trivial=True)
        {('12122222121', '12121', 'nom'): 11}
        """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree,trivial=trivial)
    
    #compute reactivation values for the whole tree
    total_reactivation = {}
    for node in IOTree.struct.values():
        total_reactivation.update(node_reactivation_plain(IOTree,node,features_set,trivial=trivial))
    return total_reactivation


################################################
# REACTIVATION FUNCTIONS: Weighted reactivation #
################################################


def extract_feature_movers(IOTree,trivial: bool=False) -> list:
    """Compute dictionary of nodes for each movement feature in a given IOTree.
        
        The dictionary is of the form {feature:[movers]}, where movers is an
        ordered list of gorn addresses associated to a specific feature.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        trivial : bool
        whether to consider intermediate movement steps
        
        Test
        -----------
        >>> tree = tree_from_file('./trees/examples/shortNOI')
        
        >>> extract_feature_movers(tree)
        {'wh': ['12121'], 'nom': ['12122222121']}
        
        >>> extract_feature_movers(tree, trivial=True)
        {'nom': ['12121', '12122222121'], 'wh': ['12121']}
        """
    
    features_movers ={}
    for node in IOTree.struct.values():
        if trivial:
            node_features = node.movement.values()
        else:
            try:
                node_features = [node.movement[next(reversed(node.movement))]]
            except:
                 node_features = []

        for feature in node_features:
            if feature in features_movers.keys():
                features_movers[feature].append(node.address)
            else:
                features_movers.update({feature:[node.address]})
    return features_movers

def node_reactivation(IOTree,IONode, features_movers, trivial: bool=False) -> dict:
    """Compute dict of reactivation values for given IONode in IOTree.
        Reactivation(m) = (1 - 1/(i(m) - o(n))). Default Reactivation is set to 1.
        
        The dictionary is of the form triple: value, where triple has the form
        (address of mover, address of previous mover, movement feature).
        
        Note that most movers will have just one value, but the dictionary is structured
        that wa for consistency with those extracted by move_length.
        
        """
    
    # only keep non-final movers if trivial is set to True
    try:
        final_target = [next(reversed(IONode.movement))]
    except:
        final_target = []
    node_features = IONode.movement if trivial else final_target

    reactivation = {}
    
    for target in node_features: #node_features.keys()
        feature = IONode.movement[target]
        #isolate list of movers per specific feature
        feature_movers = features_movers[feature]
        previous = ''
        if feature_movers.index(IONode.address) != 0:
            previous = feature_movers[feature_movers.index(IONode.address)-1]
            r =  IONode.index() - IOTree.struct[previous].outdex()
            if r <= 0:
                react = 0
            else:
                react = (1 - safediv(1,r))
            reactivation.update({(IONode.address,previous,feature):react})
        else:
            reactivation.update({(IONode.address,previous,feature):1})
    return reactivation

def reactivation_extract(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of reactivation values for all nodes/all features in  IOTree.
        See node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>>tree = tree_from_file('./trees/examples/shortNOI')
        
        >>> reactivation_extract_default(tree)
        {('12121', '', 'wh'): 1, ('12122222121', '', 'nom'): 1}
        
        >>> reactivation_extract_default(tree, trivial=True)
        {('12121', '', 'nom'): 1, ('12121', '', 'wh'): 1, ('12122222121', '12121', 'nom'): 11}
        """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree,trivial=trivial)
    
    #compute reactivation values for the whole tree
    total_reactivation = {}
    for node in IOTree.struct.values():
        total_reactivation.update(node_reactivation(IOTree,node,features_set,trivial=trivial))
    return total_reactivation

def node_boostT(IOTree,IONode, features_movers, trivial: bool=False) -> dict:
    """Compute dict of reactivation values for given IONode in IOTree.
        Reactivation(m) = Tenute(m)*(i(m) - o(n)). Default Reactivation is set to Tenure.
        
        The dictionary is of the form triple: value, where triple has the form
        (address of mover, address of previous mover, movement feature).
        
        Note that most movers will have just one value, but the dictionary is structured
        that wa for consistency with those extracted by move_length.
        
        """
    
    # only keep non-final movers if trivial is set to True
    try:
        final_target = [next(reversed(IONode.movement))]
    except:
        final_target = []
    node_features = IONode.movement if trivial else final_target

    reactivation = {}
    
    for target in node_features: #node_features.keys()
        feature = IONode.movement[target]
        #isolate list of movers per specific feature
        feature_movers = features_movers[feature]
        previous = ''
        tenure = abs(IONode.outdex() - IONode.index())
        if feature_movers.index(IONode.address) != 0:
            previous = feature_movers[feature_movers.index(IONode.address)-1]
            react_value =  IONode.index() - IOTree.struct[previous].outdex()
            if react_value <= 0:
                react_value = 0
            else:
                react_value = (1 - safediv(1,react_value))
            react = abs(tenure*react_value)
            reactivation.update({(IONode.address,previous,feature):react})
        else:
            reactivation.update({(IONode.address,previous,feature):tenure})
    return reactivation

def boostT_extract(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of reactivation values for all nodes/all features in  IOTree.
        Reactivation(m) = Tenure(m)*(i(m) - o(n)). Default Reactivation is set to Tenure.
        See node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>>tree = tree_from_file('./trees/examples/shortNOI')

        >>> reactivation_extract(tree)
        {('12121', '', 'wh'): 1, ('12122222121', '', 'nom'): 1}
        
        >>> reactivation_extract(tree, trivial=True)
        {('12121', '', 'nom'): 1, ('12121', '', 'wh'): 1, ('12122222121', '12121', 'nom'): 11}
        """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree,trivial=trivial)
    
    #compute reactivation values for the whole tree
    total_reactivation = {}
    for node in IOTree.struct.values():
        total_reactivation.update(node_boostT(IOTree,node,features_set,trivial=trivial))
    return total_reactivation



########################
# BOOST FUNCTIONS #
########################

def node_boost_div(IOTree, IONode, features_movers):
    """
        TESTS
        ------
        >>>tree = tree_from_file('./trees/examples/shortNOI')
        >>>boost_extract(tree)
   """

    #compute size and reactivation for each node
    node_size = move_length(IOTree, IONode, trivial=True)
    node_react = node_reactivation(IOTree,IONode, features_movers)
    
    features =  {}
    for movement in node_size.keys():
        features.update({movement[2]:node_size[movement]})
    
    boost = {}
    for node in node_react.keys():
        x = features[node[2]]
        y = node_react[node]
        boost_val = safediv(x,y)
        boost.update({(IONode.address,'',node[2]):boost_val})
    return boost

def boost_div_extract(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of boost values for all nodes/all features in  IOTree.
        Boost is defined as size/reactivation.
        
        see node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : True
        unused at the moment, here for consistency with tenure and size
        
        TESTS
        ------
        >>> tree = tree_from_file('./trees/examples/shortNOI')
        >>> boost_extract(tree)
        {('12121', '', 'nom'): 0.5, ('12121', '', 'wh'): 0.8333333333333334, ('12122222121', '', 'nom'): 0.36363636363636365}
        """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree)
    
    #compute reactivation values for the whole tree
    total_boost = {}
    for node in IOTree.struct.values():
        total_boost.update(node_boost_div(IOTree,node,features_set))
    return total_boost

def node_boostTS(IOTree, IONode, features_movers, trivial: bool=False):
    """
        Computes boost value for IONode based on that node size and reactivation.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size/reactivation values are to be computed
        IONode : IONode
        node in IONode for which boost needs to be computed
        features_movers: dictionary of nodes associated to movement features in IOTree
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>>tree = tree_from_file('./trees/examples/shortNOI')
        >>>boost_extract(tree)
        """
    #compute size and reactivation for each node
    node_size = move_length(IOTree, IONode, trivial=trivial)
    node_react = node_boostT(IOTree,IONode, features_movers, trivial=trivial)
    
    features =  {}
    for movement in node_size:
        features.update({movement[2]:node_size[movement]})
    
    boost = {}
    for node in node_react:
        boost.update({(IONode.address,'',node[2]):features[node[2]]*node_react[node]})
    return boost

def boostTS_extract(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of boost values for all nodes/all features in  IOTree.
        Boost is defined as size*reactivation, and reactivation is Tenure*(1-1/R)
        See node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>> tree = tree_from_file('./trees/examples/shortNOI')
        >>> boost_extract(tree)
        {('12121', '', 'wh'): 5, ('12122222121', '', 'nom'): 4}
        
        >>> boost_extract(tree, trivial=True)
        {('12121', '', 'nom'): 3, ('12121', '', 'wh'): 5, ('12122222121', '', 'nom'): 44}
        
        >>> tree = tree_from_file('trees/rc_stacked_feat/stacked_wh_that_post_oo')
        >>> boost_extract(tree, trivial=True)
        {('121', '', 'nom'): 3, ('12121212121', '', 'nom'): 135, ('1212121212222', '', 'wh'): 8, ('1212212121', '', 'nom'): 150, ('121221212222', '', 'wh'): 336}
        >>> boost_extract(tree)
        {('121', '', 'nom'): 3, ('12121212121', '', 'nom'): 135, ('1212121212222', '', 'wh'): 8, ('1212212121', '', 'nom'): 150, ('121221212222', '', 'wh'): 336}
    """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree, trivial=trivial)
    
    #compute reactivation values for the whole tree
    total_boost = {}
    for node in IOTree.struct.values():
        total_boost.update(node_boostTS(IOTree,node,features_set, trivial=trivial))
    return total_boost

###############
# Boost defined only with respect to size, so no tenure in reactivation
##############

def node_boostS(IOTree, IONode, features_movers, trivial: bool=False):
    """
        Computes boost value for IONode based on that node size and reactivation.
        Uses reactivation computed with default set to 1 instead of tenure.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size/reactivation values are to be computed
        IONode : IONode
        node in IONode for which boost needs to be computed
        features_movers: dictionary of nodes associated to movement features in IOTree
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        """
    #compute size and reactivation for each node
    node_size = move_length(IOTree, IONode, trivial=trivial)
    node_react = node_reactivation(IOTree,IONode, features_movers, trivial=trivial)
    
    features =  {}
    for movement in node_size:
        features.update({movement[2]:node_size[movement]})
    
    boost = {}
    for node in node_react:
        boost.update({(IONode.address,'',node[2]):features[node[2]]*node_react[node]})
    return boost

def boostS_extract(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of boost values for all nodes/all features in  IOTree.
        Boost is defined as size*reactivation.
        Uses reactivation computed with default set to 1 instead of tenure.
        See node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>> tree = tree_from_file('./trees/examples/shortNOI')
        >>> boost_size_extract(tree)
        {('12121', '', 'wh'): 5, ('12122222121', '', 'nom'): 4}
        
        >>> tree = tree_from_file('trees/rc_stacked_feat/stacked_wh_that_post_oo')
        >>> boost_size_extract(tree)
        {('121', '', 'nom'): 3, ('12121212121', '', 'nom'): 27, ('1212121212222', '', 'wh'): 8, ('1212212121', '', 'nom'): 30, ('121221212222', '', 'wh'): 336}
        
        >>> boost_size_extract(tree, trivial = True)
        {('121', '', 'nom'): 3, ('12121212121', '', 'nom'): 27, ('1212121212222', '', 'wh'): 8, ('1212212121', '', 'nom'): 30, ('121221212222', '', 'wh'): 336}
        
        """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree, trivial=trivial)
    
    #compute reactivation values for the whole tree
    total_boost = {}
    for node in IOTree.struct.values():
        total_boost.update(node_boostS(IOTree,node,features_set, trivial=trivial))
    return total_boost

###############
# Boost defined only with respect to size, reactivation defined as plain.
##############

def node_boost_plain(IOTree, IONode, features_movers, trivial: bool=False):
    """
        Computes boost value for IONode based on that node size and reactivation.
        Uses reactivation computed with default set to [] instead of tenure.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size/reactivation values are to be computed
        IONode : IONode
        node in IONode for which boost needs to be computed
        features_movers: dictionary of nodes associated to movement features in IOTree
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        """
    #compute size and reactivation for each node
    node_size = move_length(IOTree, IONode, trivial=trivial)
    node_react = node_reactivation_plain(IOTree,IONode, features_movers, trivial=trivial)
    
    features =  {}
    for movement in node_size:
        features.update({movement[2]:node_size[movement]})
    
    boost = {}
    for node in node_react:
        boost.update({(IONode.address,'',node[2]):features[node[2]]*node_react[node]})
    return boost

def boost_extract_plain(IOTree, filters: list=[], trivial: bool=False):
    """Compute dict of boost values for all nodes/all features in  IOTree.
        Boost is defined as size*reactivation.
        Uses reactivation computed with default set to [] instead of tenure.
        See node_reactivation for details about the format of the dict.
        
        Parameters
        ----------
        IOTree : IOTree
        index/outdex annotated Gorn tree whose size values are to be computed
        filters: list of str
        unused at the moment, here for consistency with tenure and size
        trivial : bool
        whether to consider intermediate movement steps
        
        TESTS
        ------
        >>> tree = tree_from_file('./trees/examples/shortNOI')
        >>> boost_extract_plain(tree)
        {}
        
        >>> tree = tree_from_file('trees/rc_stacked_feat/stacked_wh_that_post_oo')
        >>> boost_extract_plain(tree)
        {('12121212121', '', 'nom'): 27, ('1212212121', '', 'nom'): 30, ('121221212222', '', 'wh'): 336}
        
        >>> boost_extract_plain(tree, trivial = True)
        {('12121212121', '', 'nom'): 27, ('1212212121', '', 'nom'): 30, ('121221212222', '', 'wh'): 336}
        """
    
    #get dict of feastures and associated movers
    features_set = extract_feature_movers(IOTree, trivial=trivial)
    
    #compute reactivation values for the whole tree
    total_boost = {}
    for node in IOTree.struct.values():
        total_boost.update(node_boost_plain(IOTree,node,features_set, trivial=trivial))
    return total_boost


###################
#  Main Function  #
###################

def memory_measure(IOTree,
                   operator: 'function'=None, load_type: str='tenure',
                   filters: list=[], trivial: bool=False) -> 'int/list':
    """A general method for computing processing complexity values of IOTrees.

    With the right choice of operator and memory type, this function computes
    a variety of memory-load values for an index/outdex annotated tree.

    Metric   | Operator        | Memory Type
    ---------|-----------------|----------------
    MaxT     | safemax         | tenure_extract
    SumT     | sum             | tenure_extract
    BoxT     | len             | tenure_extract
    AvgT     | avg             | tenure_extract
    MaxTR    | None/sorted     | tenure_extract
    --------------------------------------------
    MaxS     | safemax         | move_extract
    SumS     | sum             | move_extract
    Movers   | len             | move_extract
    AvgS     | avg             | move_extract
    MaxSR    | None/sorted     | move_extract
    --------------------------------------------
    MaxR     | safemax         | reactivation_extract
    SumR     | sum             | reactivation_extract
    AvgR     | avg             | reactivation_extract
    --------------------------------------------
    MaxRp     | safemax         | reactivation_plain_extract
    SumRp     | sum             | reactivation_plain_extract
    AvgRp     | avg             | reactivation_plain_extract
    --------------------------------------------
    MaxBT     | safemax         | boostT_extract
    SumBT     | sum             | boostT_extract
    AvgBT     | avg             | boostT_extract
    --------------------------------------------
    MaxBTS     | safemax         | boostTS_extract
    SumBTS     | sum             | boostTS_extract
    AvgBTS     | avg             | boostTS_extract
    --------------------------------------------
    MaxBS     | safemax         | boostS_extract
    SumBS     | sum             | boostS_extract
    AvgBS     | avg             | boostS_extract
    --------------------------------------------
    MaxBp     | safemax         | boost_extract_plain
    SumBp     | sum             | boost_extract_plain
    AvgBp     | avg             | boost_extract_plain
    
    Parameters
    ----------
    IOTree : IOTree
        index/outdex annotated Gorn tree for which values are to be computed
    operator : function
        what function to apply to the list of tenure values;
    load_type : str
        whether to compute tenure or size-based values with tenure_extract
        or move_extract, respectively; either one returns a dictionary
    filters : list of str
        do not consider the values created by objects of a specific type;
        with load_type = tenure:
            interior (I), lexical (L), pronounced (P), unpronounced (U),
            functional (F), or content (C)
        with load_type = movement:
            names of features to ignore
    trivial : bool
        whether to include trivial instances of memory load
        with load_type = tenure:
            consider nodes with trivial tenure (= tenure 1 or 2)
        with load_type = size:
            consider instances of intermediate movement

    Examples
    --------
    >>> tree = tree_from_file('./examples/ugly')
    >>> memory_measure(ugly)
    [24, 16, 13, 12, 11, 10, 3, 3, 3]
    >>> memory_measure(ugly, load_type='tenure')
    [24, 16, 13, 12, 11, 10, 3, 3, 3]
    >>> memory_measure(ugly, load_type='size')
    [6]
    >>> memory_measure(ugly, trivial=True)
    [24, 16, 13, 12, 11, 10, 3, 3, 3, 2, 2,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> memory_measure(ugly, operator=safemax)
    24
    >>> memory_measure(ugly, operator=sum)
    95
    >>> memory_measure(ugly, operator=sum, trivial=True)
    117
    >>>tree = tree_from_file('./trees/examples/shortNOI')
    >>>memory_measure(tree, load_type='react', trivial=True)
    [11, 1, 1]
    >>>memory_measure(tree, load_type='react')
    [1, 1]
    >>>memory_measure(tree, load_type='reactP')
    []
    >>>memory_measure(tree, load_type='reactT')
    [1, 1]
    >>>memory_measure(tree, load_type='react', trivial = True)
    [11, 1, 1]
    >>>memory_measure(tree, load_type='reactP', trivial = True)
    [11]
    >>>memory_measure(tree, load_type='reactT', trivial = True)
    [11, 1, 1]
    >>>memory_measure(tree, load_type='react', operator = max)
    1
    
    >>>memory_measure(tree, load_type='boostT')
    [5, 4]
    >>>memory_measure(tree, load_type='boostT', trivial= True)
    [44, 5, 3]
    >>>memory_measure(tree, load_type='boostS')
    [5, 4]
    >>>memory_measure(tree, load_type='boostS', trivial = True)
    [44, 5, 3]
    >>>memory_measure(tree, load_type='boostP')
    []
    >>>memory_measure(tree, load_type='boostP', trivial= True)
    [44]
    >>>memory_measure(tree, load_type='boostS', operator = max)
    5
    """
    # for recursive metrics, lists should be ordered from largest to smallest
    if not operator or operator == sorted:
        operator = lambda x: sorted(x, reverse=True)

    if load_type == 'tenure':
        load_type = tenure_extract
    elif load_type == 'size':
        load_type = move_extract
    elif load_type == 'reactP':
        load_type = reactivation_plain_extract
    elif load_type == 'react':
        load_type = reactivation_extract
    elif load_type == 'boostTS':
        load_type = boostTS_extract
    elif load_type == 'boostT':
        load_type = boostT_extract
    elif load_type == 'boostS':
        load_type = boostS_extract
    elif load_type == 'boostP':
        load_type = boost_extract_plain
    elif load_type == 'boostD':
        load_type = boost_div_extract

    return operator(load_type(IOTree,
                              filters=filters,
                              trivial=trivial).values())


#reactivation_plain_extract
#reactivation_extract_default
# reactivation_extract
#boost_div_extract
#boost_extract
# boost_size_extract
#boost_extract_plain
# fixme: incorporate divergence and mtrack
