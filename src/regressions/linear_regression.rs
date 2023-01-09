use crate::utils::utils::append_column;
use nalgebra::DMatrix;

pub struct LinearRegression {
    pub coefficients: Option<Vec<f32>>,
    pub bias: Option<f32>,
}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {
            coefficients: None,
            bias: None,
        }
    }

    pub fn fit(&mut self, x: &DMatrix<f32>, y: &DMatrix<f32>) {
        let (nrows, _num_attributes) = x.shape();
        let a: DMatrix<f32> = append_column(x, vec![1.; nrows]);

        let r = a.svd(true, true).solve(y, 1.0).unwrap();

        let mut coef = DMatrix::<f32>::zeros(r.nrows(), 1);
        for i in 0..coef.nrows() - 1 {
            coef[(i, 0)] = r[(i, 0)];
        }
        self.coefficients = Some((0..r.nrows() - 1).map(|idx| r[(idx, 0)] as f32).collect());
        self.bias = Some(r[(r.nrows() - 1, 0)]);
    }

    pub fn predict(&self, x: &DMatrix<f32>) -> DMatrix<f32> {
        let values = DMatrix::from_vec(
            self.coefficients.as_ref().unwrap().len(),
            1,
            self.coefficients.as_ref().unwrap().clone(),
        );

        return (x * values).add_scalar(self.bias.as_ref().unwrap().clone());
    }
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}
