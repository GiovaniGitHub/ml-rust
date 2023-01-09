use std::vec;

use nalgebra::DMatrix;
use rand::seq::SliceRandom;
use rand::thread_rng;
// use std::default::Default;
//use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix};

pub fn expand_matrix(x: &DMatrix<f32>, degree: usize) -> DMatrix<f32> {
    let mut data_expanded = DMatrix::zeros(x.nrows(), degree);

    for row in 0..x.shape().0 {
        for col in 1..degree {
            data_expanded[(row, col - 1)] = x.data.as_vec().to_vec()[row].powf(col as f32);
        }
        data_expanded[(row, degree - 1)] = x.data.as_vec().to_vec()[row].powf(0.0);
    }
    return data_expanded;
}

pub fn train_test_split(
    x: DMatrix<f32>,
    y: DMatrix<f32>,
    test_size: f32,
    shuffle: bool,
) -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {
    if x.shape().0 != y.shape().0 {
        panic!(
            "x and y should have the same number of samples. |x|: {}, |y|: {}",
            x.shape().0,
            y.shape().0
        );
    }

    if test_size <= 0. || test_size > 1.0 {
        panic!("test_size should be between 0 and 1");
    }

    let n = y.shape().0;

    let n_test = ((n as f32) * test_size) as usize;

    if n_test < 1 {
        panic!("number of sample is too small {}", n);
    }

    let mut indices = (0..n).collect::<Vec<usize>>();

    let mut rng = thread_rng();
    if shuffle {
        indices.shuffle(&mut rng);
    }

    let x_train = slice_by_row(&x, &indices[n_test..n]);
    let x_test = slice_by_row(&x, &indices[0..n_test]);
    let y_train = slice_by_row(&y, &indices[n_test..n]);
    let y_test = slice_by_row(&y, &indices[0..n_test]);

    (x_train, x_test, y_train, y_test)
}

pub fn accuracy(y_hat: Vec<f32>, y_target: Vec<f32>) -> f32 {
    return (y_hat
        .iter()
        .enumerate()
        .map(|(k, v)| v == &y_target[k])
        .filter(|v| *v)
        .count() as f32)
        / (y_target.len() as f32);
}

pub fn flat_matrix(m: &DMatrix<f32>) -> DMatrix<f32> {
    let mut new_m = DMatrix::<f32>::zeros(m.nrows(), m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            new_m[(i, j)] = m.row(i)[j];
        }
    }

    return new_m;
}

pub fn append_column(m: &DMatrix<f32>, new_column: Vec<f32>) -> DMatrix<f32> {
    let mut new_m = DMatrix::<f32>::zeros(m.nrows(), m.ncols() + 1);
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            new_m[(i, j)] = m.row(i)[j];
        }
        new_m[(i, m.ncols())] = new_column[i];
    }

    return new_m;
}

pub fn matmul(a: &DMatrix<f32>, b: &DMatrix<f32>) -> DMatrix<f32> {
    if a.ncols() != b.nrows() {
        panic!(
            "{}",
            format!(
                "Number of rows of A ({:?}) should equal number of columns of B ({:?})",
                a.shape(),
                b.shape()
            )
        );
    }
    let inner_d = a.ncols();
    let mut result = DMatrix::zeros(a.nrows(), b.ncols());

    for r in 0..a.nrows() {
        for c in 0..b.ncols() {
            let mut s = 0.0;
            for i in 0..inner_d {
                s += a.row(r)[i] * b.row(i)[c];
            }
            result[(r, c)] = s;
        }
    }

    result
}

pub fn slice_by_row(a: &DMatrix<f32>, idx: &[usize]) -> DMatrix<f32> {
    let mut vec_values = vec![0.0; idx.len() * a.ncols()];
    for (i, row) in idx.iter().enumerate() {
        for col in 0..a.ncols() {
            vec_values[col * idx.len() + i] = a[(row.clone(), col)];
        }
    }

    return DMatrix::from_vec(idx.len(), a.ncols(), vec_values);
}
