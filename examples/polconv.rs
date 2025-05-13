//pub mod h5parm;

use clap::{ArgAction, Parser};
use ndarray::{s, ArrayD};
use num::complex::Complex;
use std::process::Command;

extern crate lofar_h5parm_rs;

/// A Rust port of the polconv functionality of h5_merger.py by Jurjen de Jong.
#[derive(Parser, Debug)]
#[command(name = "H5_polconv-rs")]
#[command(author = "Frits Sweijen")]
#[command(version = "0.0.0")]
#[command(
    help_template = "{name} \nVersion: {version} \nAuthor: {author}\n{about-section} \n {usage-heading} {usage} \n {all-args} {tab}"
)]
// #[clap(author="Author Name", version, about="")]
struct Args {
    /// H5parm to convert.
    #[arg(long)]
    h5parm: String,
    /// SolSet to convert.
    #[arg(long, default_value = "sol000")]
    solset: String,
    /// Convert from linear to circular.
    #[arg(long, action=ArgAction::SetTrue)]
    lin2circ: bool,
    /// Convert from circular to linear.
    #[arg(long, action=ArgAction::SetTrue)]
    circ2lin: bool,
    /// Convert from circular correlations R/L to Stokes parameters IQUV.
    #[arg(long, action=ArgAction::SetTrue)]
    circ2stokes: bool,
    /// [USE WITH CAUTION] Do the conversion in-place, overwriting the input solutions.
    #[arg(long, action=ArgAction::SetTrue)]
    inplace: bool,
}

fn main() {
    let args = Args::parse();
    let h5out = if args.circ2lin {
        println!("Converting solutions from circular to linear.");
        Command::new("cp")
            .args(&[&args.h5parm, &format!("{}.lin.h5", args.h5parm)])
            .status()
            .expect("Failed to create working copy of H5parm.");
        format!("{}.lin.h5", args.h5parm)
    } else if args.lin2circ {
        println!("Converting solutions from linear to circular.");
        Command::new("cp")
            .args(&[&args.h5parm, &format!("{}.circ.h5", args.h5parm)])
            .status()
            .expect("Failed to create working copy of H5parm.");
        format!("{}.circ.h5", args.h5parm)
    } else if args.circ2stokes {
        println!("Converting solutions from circular to Stokes.");
        Command::new("cp")
            .args(&[&args.h5parm, &format!("{}.stokes.h5", args.h5parm)])
            .status()
            .expect("Failed to create working copy of H5parm.");
        format!("{}.stokes.h5", args.h5parm)
    } else {
        "".to_string()
    };
    // let h5out = format!("{}.lin.h5", args.h5parm);
    println!("Reading H5parm");
    let h5parm1 = lofar_h5parm_rs::H5parm::open(&h5out, false).expect("FAILURE");
    println!("Reading SolSet");
    let solset1 = h5parm1
        .get_solset(args.solset)
        .expect("Failed to load solset");
    let soltabs = solset1.get_soltab_names();
    let mut vals_a: ArrayD<f64>;
    let mut vals_p: ArrayD<f64>;
    if soltabs.contains(&"amplitude000".to_string()) {
        println!("Reading amplitudes");
        let amp = solset1
            .get_soltab("amplitude000".to_string())
            .expect("Loading amplitude soltab failed");
        vals_a = amp.get_values();
        if !soltabs.contains(&"phase000".to_string()) {
            vals_p = ArrayD::zeros(vals_a.shape());
        } else {
            let phase = solset1
                .get_soltab("phase000".to_string())
                .expect("Loading phase soltab failed");
            vals_p = phase.get_values();
        }
    } else if soltabs.contains(&"phase000".to_string()) {
        println!("Reading phases");
        let phase = solset1
            .get_soltab("phase000".to_string())
            .expect("Loading phase soltab failed");
        vals_p = phase.get_values();
        vals_a = ArrayD::zeros(vals_p.shape());
    }else{
        panic!("No ampitude or phase soltab found, cannot convert.");
    }

    let phase = solset1
        .get_soltab("phase000".to_string())
        .expect("Loading soltab failed");
    let time = phase.get_times();
    let freq = phase.get_frequencies().unwrap();
    let ant = phase.get_antennas();
    let pols = phase.get_polarisations();
    dbg!(pols);

    let j: Complex<f64> = Complex::i();
    println!("Starting conversion");
    if args.circ2lin {
        for idx_t in 0..time.len() {
            for idx_f in 0..freq.len() {
                for idx_a in 0..ant.len() {
                    let submatrix_a = vals_a.slice(s![idx_t, idx_f, idx_a, 0, 0..4]);
                    let submatrix_p = vals_p.slice(s![idx_t, idx_f, idx_a, 0, 0..4]);

                    let g_rr = Complex::<f64>::from_polar(
                        submatrix_a[0 as usize],
                        submatrix_p[0 as usize],
                    );
                    let g_rl = Complex::<f64>::from_polar(
                        submatrix_a[1 as usize],
                        submatrix_p[1 as usize],
                    );
                    let g_lr = Complex::<f64>::from_polar(
                        submatrix_a[2 as usize],
                        submatrix_p[2 as usize],
                    );
                    let g_ll = Complex::<f64>::from_polar(
                        submatrix_a[3 as usize],
                        submatrix_p[3 as usize],
                    );
                    // Convert circular polarization to linear polarization
                    let g_xx = (g_rr + g_rl + g_lr + g_ll).unscale(2.0);
                    let g_xy = (j * g_rr - j * g_rl + j * g_lr - j * g_ll).unscale(2.0);
                    let g_yx = (-j * g_rr - j * g_rl + j * g_lr + j * g_ll).unscale(2.0);
                    let g_yy = (g_rr - g_rl - g_lr + g_ll).unscale(2.0);

                    // Convert back to new amplitudes and phases.
                    let new_amp_xx = g_xx.norm();
                    let new_amp_xy = g_xy.norm();
                    let new_amp_yx = g_yx.norm();
                    let new_amp_yy = g_yy.norm();

                    let new_phase_xx = g_xx.arg();
                    let new_phase_xy = g_xy.arg();
                    let new_phase_yx = g_yx.arg();
                    let new_phase_yy = g_yy.arg();

                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 0]] = new_amp_xx;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 1]] = new_amp_xy;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 2]] = new_amp_yx;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 3]] = new_amp_yy;

                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 0]] = new_phase_xx;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 1]] = new_phase_xy;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 2]] = new_phase_yx;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 3]] = new_phase_yy;
                }
            }
        }
    } else if args.lin2circ {
        for idx_t in 0..time.len() {
            for idx_f in 0..freq.len() {
                for idx_a in 0..ant.len() {
                    let submatrix_a = vals_a.slice(s![idx_t, idx_f, idx_a, 0, 0..4]);
                    let submatrix_p = vals_p.slice(s![idx_t, idx_f, idx_a, 0, 0..4]);

                    let g_xx = Complex::<f64>::from_polar(
                        submatrix_a[0 as usize],
                        submatrix_p[0 as usize],
                    );
                    let g_xy = Complex::<f64>::from_polar(
                        submatrix_a[1 as usize],
                        submatrix_p[1 as usize],
                    );
                    let g_yx = Complex::<f64>::from_polar(
                        submatrix_a[2 as usize],
                        submatrix_p[2 as usize],
                    );
                    let g_yy = Complex::<f64>::from_polar(
                        submatrix_a[3 as usize],
                        submatrix_p[3 as usize],
                    );
                    // Convert circular polarization to linear polarization
                    let g_rr = (g_xx - j * g_xy + j * g_yx + g_yy).unscale(2.0);
                    let g_rl = (g_xx + j * g_xy + j * g_yx - g_yy).unscale(2.0);
                    let g_lr = (g_xx - j * g_xy - j * g_yx - g_yy).unscale(2.0);
                    let g_ll = (g_xx + j * g_xy - j * g_yx + g_yy).unscale(2.0);

                    // Convert back to new amplitudes and phases.
                    let new_amp_rr = g_rr.norm();
                    let new_amp_rl = g_rl.norm();
                    let new_amp_lr = g_lr.norm();
                    let new_amp_ll = g_ll.norm();

                    let new_phase_rr = g_rr.arg();
                    let new_phase_rl = g_rl.arg();
                    let new_phase_lr = g_lr.arg();
                    let new_phase_ll = g_ll.arg();

                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 0]] = new_amp_rr;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 1]] = new_amp_rl;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 2]] = new_amp_lr;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 3]] = new_amp_ll;

                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 0]] = new_phase_rr;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 1]] = new_phase_rl;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 2]] = new_phase_lr;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 3]] = new_phase_ll;
                }
            }
        }
    } else if args.circ2stokes {
        for idx_t in 0..time.len() {
            for idx_f in 0..freq.len() {
                for idx_a in 0..ant.len() {
                    let submatrix_a = vals_a.slice(s![idx_t, idx_f, idx_a, 0, 0..4]);
                    let submatrix_p = vals_p.slice(s![idx_t, idx_f, idx_a, 0, 0..4]);

                    let g_rr = Complex::<f64>::from_polar(
                        submatrix_a[0 as usize],
                        submatrix_p[0 as usize],
                    );
                    let g_rl = Complex::<f64>::from_polar(
                        submatrix_a[1 as usize],
                        submatrix_p[1 as usize],
                    );
                    let g_lr = Complex::<f64>::from_polar(
                        submatrix_a[2 as usize],
                        submatrix_p[2 as usize],
                    );
                    let g_ll = Complex::<f64>::from_polar(
                        submatrix_a[3 as usize],
                        submatrix_p[3 as usize],
                    );
                    // Convert circular polarization to linear polarization
                    let g_i = (g_rr + g_ll).unscale(2.0);
                    let g_q = (g_rl + g_lr).unscale(2.0);
                    let g_u = -j * (g_rl - g_lr).unscale(2.0);
                    let g_v = (g_rr - g_ll).unscale(2.0);

                    // Convert back to new amplitudes and phases.
                    let new_amp_i = g_i.norm();
                    let new_amp_q = g_q.norm();
                    let new_amp_u = g_u.norm();
                    let new_amp_v = g_v.norm();

                    let new_phase_i = g_i.arg();
                    let new_phase_q = g_q.arg();
                    let new_phase_u = g_u.arg();
                    let new_phase_v = g_v.arg();

                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 0]] = new_amp_i;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 1]] = new_amp_q;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 2]] = new_amp_u;
                    *&mut vals_a[[idx_t, idx_f, idx_a, 0, 3]] = new_amp_v;

                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 0]] = new_phase_i;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 1]] = new_phase_q;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 2]] = new_phase_u;
                    *&mut vals_p[[idx_t, idx_f, idx_a, 0, 3]] = new_phase_v;
                }
            }
        }
    }
    h5parm1
        .file
        .group("/sol000/phase000")
        .expect("Failed to read group")
        .dataset("val")
        .unwrap()
        .write(&vals_p)
        .expect("Failed to write back to H5parm."); //.unwrap_or_else(|_err| panic!("Failed to read values for SolTab {}", stringify!(full_st_name)));
    h5parm1
        .file
        .group("/sol000/amplitude000")
        .unwrap()
        .dataset("val")
        .unwrap()
        .write(&vals_a)
        .expect("Failed to write back to H5parm."); //.unwrap_or_else(|_err| panic!("Failed to read values for SolTab {}", stringify!(full_st_name)));

    h5parm1.file.flush().expect("Failed to write data to file.");
    h5parm1.file.close().expect("Failed to close H5parm.");
}
