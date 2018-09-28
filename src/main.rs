use std::f32;

extern crate rand;
use rand::thread_rng;
use rand::distributions::{Normal, Distribution};

/// Generate linearly spaced vector.
/// Returns `n` evenly spaced samples, calculated over the interval [`start`, `stop`].
fn linspace(x1:f32, x2:f32, n:usize) -> Vec<f32> {
    assert!(x1 < x2, "linspace called with x1 >= x2");
    assert!(n >= 2,  "linspace called with n < 2");

    let mut y : Vec<f32> = Vec::with_capacity(n);
    
    let step = (x2 - x1) / (n - 1) as f32;

    for i in 0..n {
        y.push(x1 + step * i as f32);
    }

    y
}

#[test]
fn linspace_test() {
    assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0])
}


fn generate_moons_dataset (n_samples: usize, noise: f64) -> (Vec<(f32,f32)>, Vec<i32>) {
    assert!(n_samples >= 2, "generate_moons_dataset called with n_samples < 2");
    assert!(noise >= 0.0, "generate_moons_dataset called with noise < 0");

    let mut samples : Vec<(f32,f32)> = Vec::with_capacity(n_samples);
    let mut labels : Vec<i32> = Vec::with_capacity(n_samples);

    let mut rng = thread_rng();
    let normal = Normal::new(0., noise);

    let sc : Vec<_> = linspace(0.0, 2.0 * f32::consts::PI, n_samples).iter()
                        .map(|&x| x.sin_cos())
                        .collect();

    for i in 0 .. n_samples/2 {
        let (sin,cos) = sc[i];
        samples.push((cos + normal.sample(&mut rng) as f32, sin + normal.sample(&mut rng) as f32));

        labels.push(0);
    }
    for i in n_samples/2 .. n_samples {
        let (sin,cos) = sc[i];
        samples.push((1. + cos + normal.sample(&mut rng) as f32, 0.5 + sin + normal.sample(&mut rng) as f32));

        labels.push(1);
    }

    (samples, labels)
}


fn main() {
    // Generating Moons dataset
    let (samples, labels) = generate_moons_dataset(200, 0.20);

    // Print the data for external visualization
    for (i, sample) in samples.iter().enumerate() { 
        println!("{},{},{}", sample.0, sample.1, labels[i]);
    }
}
