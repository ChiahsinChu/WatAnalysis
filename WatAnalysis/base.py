from MDAnalysis.transformations import translate, wrap


def get_com_pos(universe, elem_type):
    elem_ag = universe.select_atoms("name " + elem_type)
    p = elem_ag.positions.copy()
    ti_com_z = p.mean(axis=0)[-1]
    return ti_com_z


def center(universe, elem_type, cell=None):
    """
    TBC
    """
    # check/set dimensions of universe
    if universe.dimensions is None:
        if cell is None:
            raise AttributeError('Cell parameter required!')
        else:
            universe.dimensions = cell
    # translate and wrap
    metal_com_z = get_com_pos(universe, elem_type)
    workflow = [translate([0, 0, -metal_com_z]), wrap(universe.atoms)]
    universe.trajectory.add_transformations(*workflow)
    #return universe