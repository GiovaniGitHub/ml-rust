pub enum TypeRegression {
    MAE,
    MSE,
    HUBER,
}

pub enum TypeFactoration {
    SVD,
    QR,
    LU,
}

pub enum Option<TypeFactoration> {
    None,
    Some(TypeFactoration),
}
