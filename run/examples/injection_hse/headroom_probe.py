"""The default sensor placement of the dataset tools (the part of the headroom probe that
``generate_se_task_datasets.py`` imports).
"""

from __future__ import annotations


#: The probe's placement on the CIGRE LV benchmark (one sensor per feeder).
CIGRE_LV_SENSORS = ("Bus R4", "Bus C5", "Bus I2")


def default_sensors(grid) -> list[str]:
    """The grid's voltage-metered node names (slack excluded), else :data:`CIGRE_LV_SENSORS`."""
    from pgml.topology import slack_node_ids

    from injection_hse.data import MeasurementModel

    if not grid.measurement_devices:
        return list(CIGRE_LV_SENSORS)
    slack = {int(s) for s in slack_node_ids(grid)}
    names = {int(nd.id): nd.name or str(nd.id) for nd in grid.nodes}
    nodes = MeasurementModel.from_grid(grid).voltage_nodes
    return [names[int(n)] for n in nodes if int(n) not in slack]


__all__ = ["CIGRE_LV_SENSORS", "default_sensors"]
