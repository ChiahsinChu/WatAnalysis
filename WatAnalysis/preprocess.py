# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


def make_selection(
    sel_region,
    surf_ids,
    c_ag="name O",
    select_all=False,
    bonded=False,
):
    """
    sel_region:
        selected region
    """
    assert len(sel_region) == 2
    assert len(surf_ids) == 2

    surf_lo = make_index_selection(surf_ids[0])
    surf_lo = "(" + surf_lo + ")"
    surf_hi = make_index_selection(surf_ids[1])
    surf_hi = "(" + surf_hi + ")"

    sel_region = np.abs(sel_region)
    lower_bound = np.min(sel_region)
    upper_bound = np.max(sel_region)

    surf_lo_region = make_relporp_selection(surf_lo, [lower_bound, upper_bound])
    surf_hi_region = make_relporp_selection(surf_hi, [-upper_bound, -lower_bound])
    select = "(" + surf_lo_region + ") or (" + surf_hi_region + ")"

    if c_ag is not None:
        select = "(" + select + ") and (%s)" % c_ag

    if select_all:
        select = "byres (" + select + ")"

    if bonded:
        select = "bonded (" + select + ")"

    return select


def make_index_selection(id_list):
    selection = "(index %d)" % id_list[0]
    for ii in id_list[1:]:
        selection = selection + " or (index %d)" % ii
    return selection


def make_relporp_selection(sel_ag, sel_region, direction="z"):
    """
    lower_bound < sel <= upper_bound
    """
    lower_bound = sel_region[0]
    upper_bound = sel_region[1]

    lo_region = "relprop %s > %f " % (direction, lower_bound) + sel_ag
    lo_region = "(" + lo_region + ")"
    hi_region = "relprop %s <= %f " % (direction, upper_bound) + sel_ag
    hi_region = "(" + hi_region + ")"

    selection = lo_region + " and " + hi_region
    return selection


def make_selection_two(
    sel_region,
    surf_ids,
    c_ag=None,
    select_all=False,
    bonded=False,
):
    """
    sel_region:
        selected region
    """
    assert len(sel_region) == 2
    assert len(surf_ids) == 2

    surf_lo = make_index_selection(surf_ids[0])
    surf_lo = "(" + surf_lo + ")"
    surf_hi = make_index_selection(surf_ids[1])
    surf_hi = "(" + surf_hi + ")"

    sel_region = np.abs(sel_region)
    lower_bound = np.min(sel_region)
    upper_bound = np.max(sel_region)

    surf_lo_region = make_relporp_selection(surf_lo, [lower_bound, upper_bound])
    surf_hi_region = make_relporp_selection(surf_hi, [-upper_bound, -lower_bound])
    select = [surf_lo_region, surf_hi_region]

    if c_ag is not None:
        select[0] = "(" + select[0] + ") and (%s)" % c_ag
        select[1] = "(" + select[1] + ") and (%s)" % c_ag

    if select_all:
        select[0] = "byres (" + select[0] + ")"
        select[1] = "byres (" + select[1] + ")"

    if bonded:
        select[0] = "bonded (" + select[0] + ")"
        select[1] = "bonded (" + select[1] + ")"

    return select
