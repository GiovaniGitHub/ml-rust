//use smartcore::linalg::{naive::dense_matrix::DenseMatrix, BaseMatrix};

//use std::ops::Sub;

use nalgebra::DMatrix;

pub fn mean(values: &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    return values.iter().sum::<f32>() / (values.len() as f32);
}

pub fn variance(values: &Vec<f32>) -> f32 {
    if values.len() == 0 {
        return 0f32;
    }

    let mean = mean(values);
    return values
        .iter()
        .map(|x| f32::powf(x - mean, 2 as f32))
        .sum::<f32>()
        / values.len() as f32;
}

pub fn covariance(x_values: &Vec<f32>, y_values: &Vec<f32>) -> f32 {
    if x_values.len() != y_values.len() {
        panic!("x_values and y_values must be of equal length.");
    }

    let length: usize = x_values.len();

    if length == 0usize {
        return 0f32;
    }

    let mut covariance: f32 = 0f32;
    let mean_x = mean(x_values);
    let mean_y = mean(y_values);

    for i in 0..length {
        covariance += (x_values[i] - mean_x) * (y_values[i] - mean_y)
    }

    return covariance / length as f32;
}

pub fn mse(y: DMatrix<f32>, y_hat: DMatrix<f32>) -> f32 {
    let n_rows = y.nrows() as f32;
    return (0..y.nrows())
        .map(|idx| ((y.row(idx)[0] - y_hat.row(idx)[0]).powf(2.0)))
        .sum::<f32>()
        / n_rows;
}

pub fn mae(y: DMatrix<f32>, y_hat: DMatrix<f32>) -> f32 {
    let n_rows = y.nrows() as f32;
    return (0..y.nrows())
        .map(|idx| ((y.row(idx)[0] - y_hat.row(idx)[0]).abs()))
        .sum::<f32>()
        / n_rows;
}

pub fn update_weights_mse_vanilla(
    x: &DMatrix<f32>,
    y: &DMatrix<f32>,
    y_hat: &DMatrix<f32>,
    lr: f32,
) -> (DMatrix<f32>, f32) {
    let (nrows, ncols) = x.shape();
    let dif = DMatrix::from_vec(nrows, 1, (y - y_hat).data.as_vec().to_vec());
    let mut dw = vec![0.0; ncols];
    for j in 0..ncols {
        for i in 0..nrows {
            dw[j] += &dif[(i, 0)] * x[(i, j)];
        }
        dw[j] *= 1.0 / (2.0 * nrows as f32) * lr;
    }
    let db = (dif.iter().sum::<f32>()) * (1.0 / (2.0 * nrows as f32)) * lr;
    return (DMatrix::from_vec(ncols, 1, dw.clone()), db);
}

pub fn update_weights_mse(
    x: &DMatrix<f32>,
    y: &DMatrix<f32>,
    y_hat: &DMatrix<f32>,
    lr: f32,
) -> (DMatrix<f32>, f32) {
    let (nrows, ncols) = x.shape();
    let dif = DMatrix::from_vec(nrows, 1, (y - y_hat).data.as_vec().to_vec());

    let dw = (0..ncols)
        .map(|i| (1.0 / (2.0 * nrows as f32)) * x.column(i).dot(&dif))
        .collect();

    let db: f32 = (dif.iter().sum::<f32>()) * (1.0 / (2.0 * nrows as f32)) * lr;
    return (DMatrix::from_vec(x.ncols(), 1, dw), db);
}

pub fn update_weights_mae(
    x: &DMatrix<f32>,
    y: &DMatrix<f32>,
    y_hat: &DMatrix<f32>,
    lr: f32,
) -> (DMatrix<f32>, f32) {
    let (nrows, ncols) = x.shape();
    let dif = DMatrix::from_vec(nrows, 1, (y - y_hat).data.as_vec().to_vec());

    let dw = (0..ncols)
        .map(|i| (1.0 / dif.abs().sum()) * x.column(i).dot(&dif) * lr)
        .collect();

    let db: f32 = (-1.0 / dif.abs().sum()) * lr * (dif.sum());
    return (DMatrix::from_vec(ncols, 1, dw), db);
}

pub fn update_weights_huber(
    x: &DMatrix<f32>,
    y: &DMatrix<f32>,
    y_hat: &DMatrix<f32>,
    lr: f32,
    delta: f32,
) -> (DMatrix<f32>, f32) {
    if (y - y_hat).abs().sum() <= delta {
        return update_weights_mae(x, y, y_hat, lr);
    } else {
        return update_weights_mse(x, y, y_hat, lr);
    };
}
