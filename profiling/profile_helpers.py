from __future__ import annotations

import numpy as np
import pandas as pd


def get_stats_dict(profile_results: list) -> list[dict[str, float]]:
    """Extract dict of results from CProfile.

    Args:
    ----
        profile_results: List of CProfile results

    Returns:
    -------
        Dict containing profiling info

    """
    stats_dicts = []
    for profile_result in profile_results:
        stats_dict = {}
        for key, item in profile_result.stats.items():
            new_key = f"{key[0]}.{key[1]}.{key[2]}"
            primitive_calls = item[0]
            calls = item[1]
            total_time = item[2]
            cumulative_time = item[3]
            stats_dict[new_key] = {
                "calls": calls,
                "calls_excl_recursion": primitive_calls,
                "time_excl_subfunctions": total_time,
                "time": cumulative_time,
                "time_per_call": total_time / calls,
            }
        stats_dicts.append(stats_dict)
    stats = [
        "calls",
        "calls_excl_recursion",
        "time_excl_subfunctions",
        "time",
        "time_per_call",
    ]

    # think the profiler will not return below a certain threshold, so the keys vary randomly,
    # find the shared set
    all_keys = [set(stats_dict.keys()) for stats_dict in stats_dicts]
    shared_keys = set.intersection(*all_keys)

    # return stats dict of avg across trials
    return {
        key: {
            stat: np.mean([stats_dict[key][stat] for stats_dict in stats_dicts])
            for stat in stats
        }
        for key in shared_keys
    }


def display_results(stats_dict: dict[str, float], stat: str) -> pd.Series:
    """Produce sorted series of runtime info for given runtime statistic.

    Args:
    ----
        stats_dict: dict of subfunction runtime info
        stat: runtime stat to look at, e.g. 'time'

    Returns:
    -------
        pd.Series: sorted series containing requested subfunction runtime info

    """
    stat_dict = {key: stats_dict[key][stat] for key in stats_dict}
    stat_series = pd.Series(stat_dict)
    return stat_series.sort_values(ascending=False)


def add_perc_stats_to_dict(
    stats_dict: dict[str, float],
    overall_time: float,
) -> dict[str, float]:
    """Add info on percentage runtime to profiling dict.

    Args:
    ----
        stats_dict: dict of stats on subfunction runtime info
        overall_time: overall runtime

    Returns:
    -------
        Dict on subfunction runtime info, with added percentage runtime info

    """
    for key, subdict in stats_dict.items():
        stats_dict[key]["percentage_time"] = (subdict["time"] / overall_time) * 100
        stats_dict[key]["percentage_time_per_call"] = (
            subdict["percentage_time"] / subdict["calls"]
        )
    return stats_dict


def profiling_wrapper(profile_results: list, display_rows: int = 10) -> None:
    """Wrapper function to display key profiling info.

    Args:
    ----
        profile_results: CProfile results
        display_rows: Top n slowest subfunctions to display. Defaults to 10.

    """
    stats_dict = get_stats_dict(profile_results)

    cumulative_time_series = display_results(stats_dict, "time")

    overall_time = cumulative_time_series.iloc[0]
    print("Overall Runtime: ", overall_time)

    stats_dict = add_perc_stats_to_dict(stats_dict, overall_time)

    perc_time_series = display_results(stats_dict, "percentage_time")
    print("Percentage Runtime: \n")
    print(perc_time_series.sort_values(ascending=False).head(display_rows))

    perc_time_per_call_series = display_results(stats_dict, "percentage_time_per_call")
    print("Percentage Runtime Per Call: \n")
    print(perc_time_per_call_series.sort_values(ascending=False).head(display_rows))
