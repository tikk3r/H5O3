//pub mod h5parm;

use std::ops::{Rem, SubAssign};

use clap::Parser;
use medians::Medianf64;
use ndarray::{s, Array1};
use num::complex::{Complex, ComplexFloat};

extern crate h5o3;

/// Flags LINC Target phase solutions based on their relative noise with respect to the core
/// stations. Solutions are flagged by setting their weight to 0. Optionally, the data can also be
/// set to NaN.
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
    /// Solset to flag.
    #[arg(long)]
    solset: String,
    /// Soltab to flag.
    #[arg(long)]
    soltab: String,
    /// Multiple of the standard deviation above which to flag solutions.
    #[arg(long, default_value = "3.0")]
    sigma: f64,
    /// Also sets the data to NaN.
    #[arg(long, default_value = "false")]
    blank_data: bool,
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
    let r = num.abs();
    assert!(r <= 1.0, "r should be smaller than one");
    assert!(r >= 0.0, "r should be larger than zero");
    (-2.0 * r.ln()).sqrt()
}

fn main() {
    let args = Args::parse();
    let h5parm = h5o3::H5parm::open(&args.h5parm, false)
        .expect("Failed opening h5parm in readwrite mode.");
    let solset = h5parm
        .get_solset(args.solset.clone())
        .expect("Failed to load solset.");
    let phase = solset
        .get_soltab(args.soltab.clone())
        .expect("Failed to load soltab.");

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
        .filter_map(|(i, &a)| a.contains("CS").then_some(i))
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
            if !detrended.is_empty() {
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
    let mut weights = phase.get_weights();

    let flag_pc_before = phase.get_flagged_fraction();

    for (station, station_name) in ant.iter().enumerate() {
        if station_name.contains("CS") || station_name.contains("RS") {
            for chan in 0..freqs.len() {
                let remaining = time.len().rem_euclid(8);
                for chunk in (0..time.len() - remaining).step_by(8) {
                    let temp_phase = vals_diff.slice(s![chunk..chunk + 8, station, chan]);
                    //if temp_phase.std(1.0) > median_std {
                    if circstd(temp_phase.to_owned()) > args.sigma * median_std {
                        vals_p
                            .slice_mut(s![chunk..chunk + 8, station, chan, ..])
                            .iter_mut()
                            .for_each(|f| *f = f64::NAN);
                        weights
                            .slice_mut(s![chunk..chunk + 8, station, chan, ..])
                            .iter_mut()
                            .for_each(|f| *f = 0.0);
                    }
                }
                let final_chunk = time.len() - remaining;
                let temp_phase = vals_diff.slice(s![final_chunk..time.len(), station, chan]);
                if circstd(temp_phase.to_owned()) > args.sigma * median_std {
                    vals_p
                        .slice_mut(s![final_chunk..time.len(), station, chan, ..])
                        .iter_mut()
                        .for_each(|f| *f = 0.0);
                    weights
                        .slice_mut(s![final_chunk..time.len(), station, chan, ..])
                        .iter_mut()
                        .for_each(|f| *f = 0.0);
                }
            }
        }
    }
    if args.blank_data {
        h5parm
            .file
            .group("/target/TGSSphase_final")
            .expect("Failed to read group")
            .dataset("val")
            .unwrap()
            .write(&vals_p)
            .expect("Failed to write values back to H5parm.");
    }
    h5parm
        .file
        .group("/target/TGSSphase_final")
        .expect("Failed to read group")
        .dataset("weight")
        .unwrap()
        .write(&weights)
        .expect("Failed to write weights back to H5parm."); //.unwrap_or_else(|_err| panic!("Failed to read values for SolTab {}", stringify!(full_st_name)));
    h5parm.file.flush().expect("Failed to write data to file.");

    let flag_pc_after = phase.get_flagged_fraction();
    println!(
        "Flagged fraction increased from {}% to {}%.",
        flag_pc_before * 100.0,
        flag_pc_after * 100.0
    );

    h5parm.file.close().expect("Failed to close H5parm.");
}
