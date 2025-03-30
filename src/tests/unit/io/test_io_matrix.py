from tidyms2.io.matrix import read_progenesis


def test_read_progenesis(data_dir):
    path = data_dir / "matrix/progenesis.csv"
    matrix = read_progenesis(path)
    assert matrix.get_n_samples() == 50
    assert matrix.get_n_features() == 100
