from src.search_space import get_search_space, Config, QUANTIZATIONS, BATCH_SIZES, PARALLELISMS

def test_search_space_size():
    space = get_search_space()
    expected = len(QUANTIZATIONS) * len(BATCH_SIZES) * len(PARALLELISMS)
    assert len(space) == expected

def test_all_configs_have_valid_fields():
    for config in get_search_space():
        assert config.q in QUANTIZATIONS
        assert config.b in BATCH_SIZES
        assert config.p in PARALLELISMS
