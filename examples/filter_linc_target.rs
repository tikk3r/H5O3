//pub mod h5parm;

use std::ops::{Rem, SubAssign};

use clap::Parser;
use medians::Medianf64;
use ndarray::{s, Array1};
use num::complex::{Complex,ComplexFloat};

extern crate lofar_h5parm_rs;

/// Flags LINC Target phase solutions based on their relative noise with respect to the core
/// stations.
#[derive(Parser, Debug)]
#[command(name = "flag-linc-target")]
#[command(author = "Frits Sweijen")]
#[command(version = "0.0.0")]
#[command(
    help_template = "{name} \nVersion: {version} \nAuthor: {author}\n{about-section} \n {usage-heading} {usage} \n {all-args} {tab}"
)]
// #[clap(author="Author Name", version, about="")]
struct Args {
    /// H5parm to flag.
    #[arg(long)]
    h5parm: String,
}

fn medfilt(input: &Array1<f64>, kernel_size: usize) -> Vec<f64> {
    assert!(kernel_size % 2 == 1, "Kernel size must be odd");

    let half_k = kernel_size / 2;
    let mut output = Vec::with_capacity(input.len());

    for i in 0..input.len() {
        let mut window = Vec::with_capacity(kernel_size);
        for j in i.saturating_sub(half_k)..=i + half_k {
            if j < input.len() {
                window.push(input[j]);
            } else {
                window.push(0.0);
            }
        }
        // Add left-side padding if necessary
        while window.len() < kernel_size {
            window.insert(0, 0.0);
        }
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        output.push(window[half_k]);
    }
    output
}

fn circstd(x: Array1<f64>) -> f64 {
    // See e.g. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circstd.html
    let mut m = Complex::new(0.0, 0.0);
    for theta in x.iter().filter(|i| i.is_finite()) {
        let z = Complex::from_polar(1.0, *theta);
        m += z;
    }
    let num = Complex::from_polar(m.abs() / x.len() as f64, m.arg());
    let R = num.abs();
    assert!(R <= 1.0, "R should be smaller than one");
    assert!(R >= 0.0, "R should be larger than zero");
    let s = (-2.0 * R.ln()).sqrt();
    s
}

fn main() {
    let args = Args::parse();
    println!("Reading H5parm");
    let h5parm1 = lofar_h5parm_rs::H5parm::open(&args.h5parm, false).expect("FAILURE");
    println!("Reading SolSet");
    let solset1 = h5parm1
        .get_solset("target".to_string())
        .expect("Failed to load solset");
    println!("Reading phases");
    let phase = solset1
        .get_soltab("TGSSphase_final".to_string())
        .expect("Loading phase soltab failed");
    let mut vals_p = phase.get_values();
    for i in 0..vals_p.shape()[1] {
        let ref_phase = vals_p.slice(s![.., 13, .., ..]).to_owned();
        vals_p.slice_mut(s![.., i, .., ..]).sub_assign(&ref_phase);
    }

    let mut vals_diff = vals_p.slice(s![.., .., .., 0]).clone().to_owned();
    vals_diff = vals_diff - vals_p.slice(s![.., .., .., 1]);
    vals_diff =
        (vals_diff + std::f64::consts::PI).rem(2.0 * std::f64::consts::PI) - std::f64::consts::PI;
    let ant = phase.get_antennas();
    let cs_idx: Vec<_> = ant
        .iter()
        .enumerate()
        .filter_map(|(i, &a)| a.contains("CS").then(|| i))
        .collect();
    let cs_scatters: Vec<f64> = cs_idx
        .iter()
        .map(|ant| {
            let temp_phase = vals_diff.slice(s![.., *ant, 0]).to_owned();
            let temp_phase = temp_phase.map(|x| if !x.is_finite() { 0.0 } else { *x });

            let filtered = Array1::from(medfilt(&temp_phase, 59));

            let detrended = temp_phase - filtered;
            let detrended: Vec<f64> = detrended.into_iter().filter(|x| x.is_finite()).collect();
            let detrended = Array1::from_vec(detrended);
            if detrended.len() > 0 {
                let scatter = circstd(detrended);
                println!("Scatter for antenna {} is {}", *ant, scatter);
                scatter
            } else {
                println!("Scatter for antenna {} is {}", *ant, 0.0);
                0.0
            }
        })
        .collect();
    let median_std = cs_scatters.medf_unchecked();
    println!("Median core scatter: {}", median_std);
    let freqs = phase.get_frequencies().unwrap();
    let time = phase.get_times();
    for chan in 0..freqs.len() {
        for (station, station_name) in ant.iter().enumerate() {
            if station_name.contains("CS") || station_name.contains("RS") {
                //if station_name.contains("CS101HBA1") {
                //print!("Channel {: >4}; {: >10}\r", chan, station_name);
                let remaining = time.len().rem_euclid(8);
                for chunk in (0..time.len() - remaining).step_by(8) {
                    let temp_phase = vals_diff.slice(s![chunk..chunk + 8, station, chan]);
                    if temp_phase.std(1.0) > median_std {
                        vals_p
                            .slice_mut(s![chunk..chunk + 8, station, chan, ..])
                            .iter_mut()
                            .for_each(|f| *f = std::f64::NAN);
                    }
                }
                let final_chunk = time.len() - remaining;
                let temp_phase = vals_diff.slice(s![final_chunk..time.len(), station, chan]);
                if temp_phase.std(1.0) > median_std {
                    vals_p
                        .slice_mut(s![final_chunk..time.len(), station, chan, ..])
                        .iter_mut()
                        .for_each(|f| *f = std::f64::NAN);
                }
            }
        }
    }

    //h5parm1
    //    .file
    //    .group("/target/TGSSphase_final")
    //    .expect("Failed to read group")
    //    .dataset("val")
    //    .unwrap()
    //    .write(&vals_p)
    //    .expect("Failed to write back to H5parm."); //.unwrap_or_else(|_err| panic!("Failed to read values for SolTab {}", stringify!(full_st_name)));
    //h5parm1.file.flush().expect("Failed to write data to file.");

    h5parm1.file.close().expect("Failed to close H5parm.");
}
