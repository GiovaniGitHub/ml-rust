use crate::utils::{
    stats::{update_weights_huber, update_weights_mae, update_weights_mse},
    types::TypeRegression,
    utils::expand_matrix,
};
use nalgebra::DMatrix;
use rand::Rng;

pub struct PolynomialRegression {
    pub coefficients: DMatrix<f32>,
    pub bias: f32,
    pub degree: usize,
    pub type_regression: TypeRegression,
}

impl PolynomialRegression {
    pub fn new(degree: usize, type_regression: TypeRegression) -> PolynomialRegression {
        PolynomialRegression {
            coefficients: DMatrix::from_vec(
                degree,
                1,
                (0..degree)
                    .map(|_| rand::thread_rng().gen::<f32>())
                    .collect(),
            ),
            bias: 0.0,
            degree,
            type_regression,
        }
    }

    pub fn predict(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let expanded_matrix = if x.shape().1 != self.degree {
            expand_matrix(x, self.degree)
        } else {
            x.clone()
        };
        let mut y_hat: Vec<f32> = Vec::new();
        for i in 0..expanded_matrix.nrows() {
            y_hat.push(
                (0..expanded_matrix.ncols())
                    .map(|j| expanded_matrix[(i, j)] * self.coefficients[(j, 0)])
                    .sum(),
            );

            y_hat[i] += self.bias;
        }

        return DMatrix::from_vec(x.nrows(), 1, y_hat);
    }

    pub fn fit(&mut self, x: &DMatrix<f32>, y: &DMatrix<f32>, epochs: usize, lr: f32) {
        let expanded_matrix = expand_matrix(x, self.degree);

        match self.type_regression {
            TypeRegression::MSE => {
                for _ in 0..epochs {
                    let y_hat = self.predict(&x);
                    let (dw, db) = update_weights_mse(&expanded_matrix, &y, &y_hat, lr);
                    self.coefficients += dw;
                    self.bias += db;
                }
            }
            TypeRegression::MAE => {
                for _ in 0..epochs {
                    let y_hat = self.predict(&x);
                    let (dw, db) = update_weights_mae(&expanded_matrix, &y, &y_hat, lr);
                    self.coefficients += dw;
                    self.bias += db;
                }
            }
            TypeRegression::HUBER => {
                for _ in 0..epochs {
                    let y_hat = self.predict(&x);
                    let (dw, db) = update_weights_huber(&expanded_matrix, &y, &y_hat, lr, 1.0);
                    self.coefficients += dw;
                    self.bias += db;
                }
            }
        }
    }
}
