import narwhals as nw


def cap_columns(columns: list[str], capping_values_for_transform):

    return [
        nw.col(col).clip(
            lower_bound=capping_values_for_transform[col][0],
            upper_bound=capping_values_for_transform[col][1],
        )
        for col in columns
    ]


def set_out_of_range_to_none(columns, capping_values_for_transform):

    return [
        nw.when(nw.col(col) < (cap_value_min := capping_values_for_transform[col][0]))
        .then(None)
        .otherwise(
            nw.when(
                nw.col(col) > (cap_value_max := capping_values_for_transform[col][1])
            )
            .then(None)
            .otherwise(nw.col(col))
        )
        if cap_value_min and cap_value_max
        else nw.when(nw.col(col) < cap_value_min).then(None).otherwise(nw.col(col))
        if cap_value_min
        else nw.when(nw.col(col) > cap_value_max).then(None).otherwise(nw.col(col))
        if cap_value_max
        else nw.col(col)
        for col in columns
    ]
