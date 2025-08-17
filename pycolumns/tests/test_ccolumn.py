import pytest


@pytest.mark.parametrize("dtype", ["i8", "f4", "U3"])
def test_column(dtype):
    import os
    import tempfile
    import numpy as np
    from .._column import Column

    seed = 333
    num = 20

    rng = np.random.RandomState(seed)

    if dtype[0] == "i":
        data = np.arange(num, dtype=dtype)
    elif dtype[0] == "U":
        data = np.zeros(num, dtype=dtype)
        data[:] = [str(i) for i in range(num)]
    else:
        data = rng.uniform(size=num).astype(dtype)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, "test.col")
        with Column(fname, dtype=data.dtype, mode="w+", verbose=True) as col:
            print("-" * 70)
            print("before append")
            print(col)

            col.append(data)

            print("-" * 70)
            print("before append")
            print(col)

            inner_data = col[:]
            assert np.all(inner_data == data)

            inner_data = col[2:8]
            assert np.all(inner_data == data[2:8])

            inner_data = col[2:18:2]
            assert np.all(inner_data == data[2:18:2])

            indices = [3, 5, 7]
            inner_data = col[indices]
            assert np.all(inner_data == data[indices])

            # out of order
            indices = [7, 3, 5]
            inner_data = col[indices]
            assert np.all(inner_data == data[indices])

            indices = 5
            inner_data = col[indices]
            assert np.all(inner_data == data[indices])

            s = slice(2, 8)
            inner_data = np.zeros(s.stop - s.start, dtype=data.dtype)
            col.read_slice_into(inner_data, s)
            assert np.all(inner_data == data[s])

            s = slice(-3, -1)
            inner_data = np.zeros(2, dtype=data.dtype)
            col.read_slice_into(inner_data, s)
            assert np.all(inner_data == data[s])

            indices = [3, 5, 7]
            inner_data = np.zeros(len(indices), dtype=data.dtype)
            col.read_rows_into(inner_data, indices)
            assert np.all(inner_data == data[indices])

            indices = 6
            inner_data = np.zeros(1, dtype=data.dtype)
            col.read_row_into(inner_data, indices)
            assert np.all(inner_data[0] == data[indices])

            # cannot read into array scalar
            with pytest.raises(ValueError):
                indices = 6
                inner_data = np.zeros(1, dtype=data.dtype)
                col.read_row_into(inner_data[0], indices)

            # mismatch in ind size and data size
            with pytest.raises(ValueError):
                indices = [3, 5, 7]
                inner_data = np.zeros(5, dtype=data.dtype)
                col.read_rows_into(inner_data, indices)

            # mismatch in ind size and data size
            with pytest.raises(ValueError):
                indices = 3
                inner_data = np.zeros(5, dtype=data.dtype)
                col.read_row_into(inner_data, indices)

            col.append(data)
            assert col.nrows == data.size * 2

            #
            # updating column data
            #

            one = data[:1]
            row2update = 3
            col.update_row(row2update, one)
            check = col[3]
            assert check == one[0]

            two = data[3 : 3 + 2]
            start = 15
            col.write_at(two, start)
            check = col[start : start + 2]
            assert np.all(check == two)

            col[5] = data[0]
            assert np.all(col[5] == data[0])

            col[: data.size] = data
            assert np.all(col[: data.size] == data)

            # sorted rows
            rows = [0, 2, 5]
            wdata = np.array([99, 88, 77], dtype=dtype)
            col[rows] = wdata
            assert np.all(col[rows] == wdata)

            # out of order rows
            rows = [8, 3, 7]
            wdata = np.array([100, 102, 103], dtype=dtype)
            col[rows] = wdata
            assert np.all(col[rows] == wdata)

            rows = slice(0, 6, 2)
            wdata = np.arange(0, 6, 2).astype(dtype)
            col[rows] = wdata
            assert np.all(col[rows] == wdata)

            # bad slice size
            with pytest.raises(IndexError):
                col[0:5] = wdata

            # out of bounds update
            with pytest.raises(IndexError):
                col[np.arange(100)] = np.arange(100)
