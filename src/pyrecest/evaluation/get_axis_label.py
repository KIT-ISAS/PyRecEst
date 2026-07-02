def get_axis_label(manifold_name):
    if not isinstance(manifold_name, str) or not manifold_name.strip():
        raise ValueError("Mode not recognized")

    normalized_name = manifold_name.strip().lower()

    if "circlesymm" in normalized_name:
        error_label = "Error in radian"

    elif "circle" in normalized_name or "hypertorus" in normalized_name:
        error_label = "Error in radian"

    elif (
        "hyperspheresymmetric" in normalized_name
        or "hypersphere_symmetric" in normalized_name
    ):
        error_label = "Angular error in radian"

    elif "hypersphere" in normalized_name:
        error_label = "Error (orthodromic distance) in radian"

    elif "se2bounded" in normalized_name:
        error_label = "Error in radian"

    elif "se2" in normalized_name or "se2linear" in normalized_name:
        error_label = "Error in meters"

    elif "se3bounded" in normalized_name:
        error_label = "Error in radian"

    elif "se3" in normalized_name or "se3linear" in normalized_name:
        error_label = "Error in meters"

    elif "euclidean" in normalized_name and "mtt" not in normalized_name:
        error_label = "Error in meters"

    elif "euclidean" in normalized_name and "mtt" in normalized_name:
        error_label = "OSPA error in meters"

    else:
        raise ValueError("Mode not recognized")

    return error_label
