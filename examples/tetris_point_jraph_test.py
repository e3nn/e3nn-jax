from examples.tetris_point_jraph import train


def test_tetris_point():
    train(seeds=1, steps=50, plot=False)
