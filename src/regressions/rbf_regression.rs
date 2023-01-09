use nalgebra::DMatrix;
use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::utils::types::TypeFactoration;
use crate::utils::utils::{expand_matrix, matmul};

pub struct RBFRegression {
    pub num_center: usize,
    pub centers: DMatrix<f32>,
    pub beta: f32,
    pub weight: DMatrix<f32>,
    pub type_factoration: Option<TypeFactoration>,
}

impl RBFRegression {
    pub fn new(
        beta: f32,
        num_center: usize,
        num_cols: usize,
        type_factoration: Option<TypeFactoration>,
    ) -> RBFRegression {
        let mut coefficients_centers: Vec<f32> = Vec::new();
        let mut coefficients_weight: Vec<f32> = Vec::new();
        for _ in 0..num_center {
            coefficients_weight.push(1.0);
            for _ in 0..num_cols {
                coefficients_centers.push(1.0);
            }
        }

        RBFRegression {
            num_center,
            centers: DMatrix::from_vec(num_center, num_cols, coefficients_centers),
            beta,
            weight: DMatrix::from_vec(num_center, 1, coefficients_weight),
            type_factoration,
        }
    }

    pub fn fit(&mut self, x: &DMatrix<f32>, y: &DMatrix<f32>) {
        let (_, n_columns) = self.centers.shape();
        let x = expand_matrix(&x, n_columns);

        let (num_rows, num_cols) = x.shape();

        let mut index: Vec<usize> = (0..num_rows).collect();

        index.shuffle(&mut thread_rng());
        let mut count = 0;

        for i in index {
            for j in 0..num_cols {
                self.centers[(count, j)] = x[(i, j)];
            }
            count = count + 1;
            if count >= self.num_center {
                break;
            }
        }

        let gradient = calculate_gradient(&x, &self.centers, &self.beta);

        match self.type_factoration {
            Some(TypeFactoration::SVD) => {
                self.weight = matmul(&gradient.transpose(), &gradient)
                    .svd(true, true)
                    .solve(&matmul(&gradient.transpose(), &y), 1.0)
                    .unwrap();
            }
            Some(TypeFactoration::QR) => {
                self.weight = matmul(&gradient.transpose(), &gradient)
                    .qr()
                    .solve(&matmul(&gradient.transpose(), &y))
                    .unwrap();
            }
            Some(TypeFactoration::LU) => {
                self.weight = matmul(&gradient.transpose(), &gradient)
                    .lu()
                    .solve(&matmul(&gradient.transpose(), &y))
                    .unwrap();
            }
            _ => {
                self.weight = matmul(&gradient.transpose(), &gradient)
                    .svd(true, true)
                    .solve(&matmul(&gradient.transpose(), &y), 1.0)
                    .unwrap();
            }
        }
    }

    pub fn predict(&mut self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let (_, n_columns) = self.centers.shape();
        let x = expand_matrix(&x, n_columns);

        return matmul(
            &calculate_gradient(&x, &self.centers, &self.beta),
            &self.weight,
        );
    }
}

pub fn calculate_gradient(x: &DMatrix<f32>, centers: &DMatrix<f32>, beta: &f32) -> DMatrix<f32> {
    let mut gradient_vector: Vec<f32> = vec![0.0; x.nrows() * centers.nrows()];
    for row in 0..x.nrows() {
        for col in 0..centers.nrows() {
            let norm = (centers.row(col) - x.row(row))
                .iter()
                .map(|value| value.powf(2.0))
                .sum::<f32>();
            gradient_vector[col * x.nrows() + row] = (-beta * norm).exp();
        }
    }

    return DMatrix::from_vec(x.nrows(), centers.nrows(), gradient_vector);
}
