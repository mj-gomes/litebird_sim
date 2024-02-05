def add_convolved_sky_to_observations(
    obs_list: List[Observation],
    slm_dictionary: Dict[str, Any],  # unconvolved sky a_lm
    blm_dictionary: Dict[str, Any],  # beam a_lm
    det2slm: Dict[str, str],  # detector name -> slm name
    det2blm: Dict[str, str],  # detector name -> blm name (could be identity) 
    component: str = "tod",
):
    """Convolve sky maps with generic detector beams and add the resulting
    signal to TOD.

    Arguments
    ---------
    obs_list: List[Observation],
        List of Observation objects, containing detector names, pointings,
        and TOD data, to which the computed TOD are added.
    slm_dictionary:  Dict[str, Any]
        sky a_lm. Typically only one set of sky a_lm is needed per detector frequency
    blm_dictionary: Dict[str, Any]
        beam a_lm. Usually one set of a_lm is needed for every detector.
    det2slm: Dict[str, str]
        converts detector name to a key for `slm_dictionary`
    det2slm: Dict[str, str]
        converts detector name to a key for `blm_dictionary`
    component: str
        name of the TOD component to which the computed data shall be added
    """
    # find all involved detector names
    # ??? How do I extract detecor names from observation objects?
    for cur_det in det_list:
        # set up convolver for slm_dictionary[det2slm[cur_det]] and blm_dictionary[det2blm[cur_det]]
        for cur_obs in obs_list:
#            obs_idx = ??? how do I get the idx for the current detector?
            ptg = cur_obs.pointings[obs_idx]
            psi = cur_obs.psi[obs_idx]
            cur_tod = getattr(cur_obs, component)[obs_idx]
            cur_tod += convolver.convolve(ptg, psi)
