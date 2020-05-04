#tests

    try:
        final_target = [next(reversed(IONode.movement))]
    except:
        final_target = []
    move_targets = IONode.movement if trivial else final_target

return {(IONode.address, target, IONode.movement[target]):
    abs(IONode.index() - IOTree.struct[target].index())
    for target in move_targets
        if IONode.movement[target] not in filters}





########################
# REACTIVATION FUNCTIONS FIRST VERSION#
########################


def extract_feature_movers(IOTree,feature) -> list:
    """Get ordered list of movers associated to a specific feature"""
    
    feature_movers = []
    for node in IOTree.struct.values():
        if feature in node.movement.values():
            feature_movers.append(node.address)
    return feature_movers

def node_reactivation(IOTree, IONode) -> dict:
    """Compute dict of reactivation values for given IONode in IOTree.
        
        The dictionary is of the form triple: value, where triple has the form
        (address of mover, address of previous mover, movement feature).
        
        Note that most movers will have just one value, but the dictionary is structured
        that wa for consistency with those extracted by move_length.
        """
    node_features = IONode.movement
    reactivation = {}
    for target in node_features.keys():
        feature = IONode.movement[target]
        feature_movers = extract_feature_movers(IOTree,feature)
        previous = ''
        if feature_movers.index(IONode.address) != 0:
            previous = feature_movers[feature_movers.index(IONode.address)-1]
            react_value =  abs(IONode.index() - IOTree.struct[previous].outdex())
            reactivation.update({(IONode.address,previous,feature):react_value})
        else:
            reactivation.update({(IONode.address,previous,feature):abs(IONode.index())})
    return reactivation

def reactivation_extract(IOTree, filters: list=[], trivial: bool=False):
    total_reactivation = {}
    for node in IOTree.struct.values():
        total_reactivation.update(node_reactivation(IOTree, node))
    return total_reactivation

################################
