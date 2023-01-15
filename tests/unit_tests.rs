use nalgebra::DMatrix;
use rust_regressions::utils::utils::slice_by_row;

pub fn get_dmatrix() -> DMatrix<f32> {
    return DMatrix::from_row_slice(
        4,
        3,
        &[
            1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0,
        ],
    );
}

#[test]
fn test_dimension_slice_by_row() {
    let input = get_dmatrix();

    assert_eq!(slice_by_row(&input, &[0, 2]).shape(), (2, 3));
}
#[test]
fn test_data_slice_by_row() {
    let input = get_dmatrix();
    let expected_dmatrix = slice_by_row(&input, &[0, 2]);

    assert_eq!(slice_by_row(&input, &[0, 2]).eq(&expected_dmatrix), true);
}
