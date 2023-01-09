use nalgebra::DMatrix;
use rust_regressions::clusters::knn::KNN;
use rust_regressions::regressions::linear_regression::LinearRegression;
use rust_regressions::regressions::polynomial_regression::PolynomialRegression;
use rust_regressions::regressions::rbf_regression::RBFRegression;
use rust_regressions::regressions::simple_linear_regression::SimpleLinearRegression;
use rust_regressions::utils::io::{line_and_scatter_plot, parse_csv};

use rust_regressions::utils::types::{TypeFactoration, TypeRegression};
use rust_regressions::utils::utils::{accuracy, train_test_split};

use std::env;
use std::{fs::File, io::BufReader};

static MSG: &str = "cargo run linear|simple|poly linear_regression|simple_linear_regression|polynomial_regression_data";

fn main() {
    env::set_var("RUST_BACKTRACE", "full");
    let args: Vec<String> = env::args().collect();
    assert_eq!(args.len(), 3, "{}", MSG);

    let dataset_name_file = &args[2];
    let type_regression = &args[1];

    if "simple" == type_regression {
        let file: File = File::open(format!("datasets/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let dense_matrix = DMatrix::from_row_slice(tuple_result.0, tuple_result.1, &tuple_result.2);

        let x = dense_matrix.column(0).iter().map(|v| v.clone()).collect();
        let y = dense_matrix.column(1).iter().map(|v| v.clone()).collect();

        let mut model = SimpleLinearRegression::new();
        model.fit(&x, &y);

        let y_predictions: Vec<f32> = model.predict_list(&x);

        line_and_scatter_plot(x, vec![y, y_predictions], vec!["original", "predicted"]);
    }
    if "linear" == type_regression {
        let file: File = File::open(format!("datasets/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let mut x = DMatrix::zeros(tuple_result.0, tuple_result.1 - 1);
        let mut y = DMatrix::zeros(tuple_result.0, 1);
        for i in 0..tuple_result.0 {
            for j in 0..(tuple_result.1 - 1) {
                x[(i, j)] = tuple_result.2[i * (tuple_result.1) + j];
            }
            y[(i, 0)] = tuple_result.2[tuple_result.1 * (i + 1) - 1]
        }

        let mut model = LinearRegression::new();
        model.fit(&x, &y);

        let y_predictions = model.predict(&x);
        let y_plot = vec![
            y.data.as_vec().to_vec(),
            y_predictions.data.as_vec().to_vec(),
        ];

        line_and_scatter_plot(
            (0..x.shape().0).map(|v| v as f32).collect(),
            y_plot,
            vec!["original", "predicted"],
        );
    }
    if "poly" == type_regression {
        let file: File = File::open(format!("datasets/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let mut x = DMatrix::zeros(tuple_result.0, tuple_result.1 - 1);
        let mut y = DMatrix::zeros(tuple_result.0, 1);
        for i in 0..tuple_result.0 {
            for j in 0..(tuple_result.1 - 1) {
                x[(i, j)] = tuple_result.2[i * (tuple_result.1) + j];
            }
            y[(i, 0)] = tuple_result.2[tuple_result.1 * (i + 1) - 1]
        }

        let mut model_mse = PolynomialRegression::new(8, TypeRegression::MSE);
        model_mse.fit(&x, &y, 1000, 0.7);
        let y_hat_mse = model_mse.predict(&x);

        let mut model_mae = PolynomialRegression::new(8, TypeRegression::MAE);
        model_mae.fit(&x, &y, 1000, 0.7);
        let y_hat_mae = model_mae.predict(&x);

        let mut model_huber = PolynomialRegression::new(8, TypeRegression::HUBER);
        model_huber.fit(&x, &y, 3000, 0.7);
        let y_hat_huber = model_huber.predict(&x);

        let mut model_rbf = RBFRegression::new(4.0, 22, 8, None);
        model_rbf.fit(&x, &y);

        let y_hat_rbf = model_rbf.predict(&x);

        let y_plot = vec![
            y.data.as_vec().to_vec(),
            y_hat_mse.data.as_vec().to_vec(),
            y_hat_mae.data.as_vec().to_vec(),
            y_hat_huber.data.as_vec().to_vec(),
            y_hat_rbf.data.as_vec().to_vec(),
        ];

        line_and_scatter_plot(
            (0..x.shape().0).map(|v| v as f32).collect(),
            y_plot,
            vec!["original", "MSE", "MAE", "HUBER", "RBF"],
        );
    }
    if type_regression == "rbf" {
        let file: File = File::open(format!("datasets/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();
        let mut x = DMatrix::zeros(tuple_result.0, tuple_result.1 - 1);
        let mut y = DMatrix::zeros(tuple_result.0, 1);
        for i in 0..tuple_result.0 {
            for j in 0..(tuple_result.1 - 1) {
                x[(i, j)] = tuple_result.2[i * (tuple_result.1) + j];
            }
            y[(i, 0)] = tuple_result.2[tuple_result.1 * (i + 1) - 1]
        }

        let mut model_rbf_lu = RBFRegression::new(4.0, 24, 12, Some(TypeFactoration::LU));
        model_rbf_lu.fit(&x, &y);
        let mut model_rbf_qr = RBFRegression::new(4.0, 24, 12, Some(TypeFactoration::QR));
        model_rbf_qr.fit(&x, &y);
        let mut model_rbf_svd = RBFRegression::new(4.0, 24, 12, Some(TypeFactoration::SVD));
        model_rbf_svd.fit(&x, &y);

        let y_hat_rbf_lu = model_rbf_lu.predict(&x);
        let y_hat_rbf_qr = model_rbf_qr.predict(&x);
        let y_hat_rbf_svd = model_rbf_svd.predict(&x);

        let y_plot = vec![
            y.data.as_vec().to_vec(),
            y_hat_rbf_lu.data.as_vec().to_vec(),
            y_hat_rbf_qr.data.as_vec().to_vec(),
            y_hat_rbf_svd.data.as_vec().to_vec(),
        ];

        line_and_scatter_plot(
            (0..x.shape().0).map(|v| v as f32).collect(),
            y_plot,
            vec!["original", "LU", "QR", "SVD"],
        );
    }
    if type_regression == "knn" {
        let file: File = File::open(format!("datasets/{}.csv", dataset_name_file)).unwrap();
        let tuple_result: (usize, usize, Vec<f32>) = parse_csv(BufReader::new(file)).unwrap();

        let mut x = DMatrix::zeros(tuple_result.0, tuple_result.1 - 1);
        let mut y = DMatrix::zeros(tuple_result.0, 1);
        for i in 0..tuple_result.0 {
            for j in 0..(tuple_result.1 - 1) {
                x[(i, j)] = tuple_result.2[i * (tuple_result.1) + j];
            }
            y[(i, 0)] = tuple_result.2[tuple_result.1 * (i + 1) - 1]
        }

        let (x_train, x_test, y_train, y_test) = train_test_split(x, y, 0.5, true);

        let mut model = KNN::new(x_train, y_train, 5);

        let y_hat = model
            .predict(x_test)
            .iter()
            .map(|v| v.parse::<f32>().unwrap())
            .collect::<Vec<f32>>();
        println!(
            "Accuracy: {}",
            accuracy(
                y_hat,
                (0..y_test.nrows())
                    .map(|idx| y_test[(idx, 0)])
                    .collect::<Vec<f32>>()
            )
        );
    }
}
