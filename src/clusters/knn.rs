use std::collections::HashMap;

use nalgebra::DMatrix;

pub struct KNN {
    pub x: DMatrix<f32>,
    pub y: DMatrix<f32>,
    pub n_neighborhood: usize,
}

pub struct NeighborhoodItem {
    pub dist: f32,
    pub class: f32,
}

impl NeighborhoodItem {
    pub fn new() -> NeighborhoodItem {
        NeighborhoodItem {
            dist: -1.0,
            class: f32::MAX,
        }
    }
}

impl KNN {
    pub fn new(x: DMatrix<f32>, y: DMatrix<f32>, n_neighborhood: usize) -> KNN {
        KNN {
            x,
            y,
            n_neighborhood,
        }
    }

    pub fn predict(&mut self, x: DMatrix<f32>) -> Vec<String> {
        let mut y_hat = Vec::new();
        for idx in 0..x.nrows() {
            let mut neighborhood_list = Vec::with_capacity(self.x.shape().0);
            for i in 0..self.x.shape().0 {
                let dist: f32 = (self.x.row(i) - x.row(idx))
                    .iter()
                    .map(|r| r.powf(2.0))
                    .sum();
                let mut neighborhood_item = NeighborhoodItem::new();

                neighborhood_item.dist = dist;
                neighborhood_item.class = self.y[(i, 0)];

                neighborhood_list.push(neighborhood_item);
            }

            neighborhood_list
                .sort_by(|a: &NeighborhoodItem, b| a.dist.partial_cmp(&b.dist).unwrap());

            let mut m: HashMap<String, usize> = HashMap::new();
            for ii in 0..self.n_neighborhood {
                *m.entry(neighborhood_list[ii].class.to_string())
                    .or_default() += 1;
            }

            y_hat.push(m.iter().max_by_key(|entry| entry.1).unwrap().0.clone());
        }

        return y_hat;
    }
}
