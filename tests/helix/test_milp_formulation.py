from src.helix.milp_formulation import GPUNode, NetworkLink, solve_max_flow_placement

NODES = [
    GPUNode(node_id="gpu0", throughput_gbs=100.0, memory_gb=80.0),
    GPUNode(node_id="gpu1", throughput_gbs=60.0, memory_gb=40.0),
]
LINKS = [
    NetworkLink(src="source", dst="gpu0", bandwidth_gbs=50.0),
    NetworkLink(src="source", dst="gpu1", bandwidth_gbs=50.0),
    NetworkLink(src="gpu0", dst="sink", bandwidth_gbs=50.0),
    NetworkLink(src="gpu1", dst="sink", bandwidth_gbs=50.0),
]


def test_solve_returns_nonnegative_flow():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    assert result.total_flow >= 0


def test_solve_status_is_optimal():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    assert result.status == "Optimal"


def test_all_layers_are_placed():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    assert len(result.layer_placement) == 4


def test_layers_placed_on_valid_nodes():
    result = solve_max_flow_placement(NODES, LINKS, num_layers=4)
    valid_node_ids = {"gpu0", "gpu1"}
    for node_id in result.layer_placement.values():
        assert node_id in valid_node_ids
