#![allow(non_snake_case)]
// H5parm interface.

use anyhow::bail;
use hdf5::file;
use ndarray::{array,Array1, ArrayD};
use thiserror::Error;

pub struct H5parm {
    pub name: String,
    pub file: file::File,
    pub solsets: Vec<SolSet>,
}

impl H5parm {
    pub fn open(h5parm_in: &String, readonly: bool) -> Result<Self, Box<dyn std::error::Error>> {
        let infile = if readonly {
            file::File::open(&h5parm_in)
        } else {
            file::File::open_rw(&h5parm_in)
        };
        let solsets = infile
            .clone()?
            .groups()
            .expect("Failed to read SolSets from H5parm.");

        let mut solsetlist: Vec<SolSet> = vec![];
        for ss in solsets.iter() {
            // if ss.name().chars().nth(0).unwrap() == '/' {
            if ss.clone().name().starts_with('/') {
                let x = SolSet::init(&infile.clone()?, ss.name()[1..].to_string());
                solsetlist.push(x?);
            } else {
                let x = SolSet::init(&infile.clone()?, ss.name());
                solsetlist.push(x?);
            }
        }

        return Ok(H5parm {
            name: h5parm_in.to_string(),
            file: infile?,
            solsets: solsetlist,
        });
    }

    pub fn getSolSet(&self, ssname: String) -> &SolSet {
        let index = self.solsets.iter().position(|r| r.name == ssname).unwrap();
        return &self.solsets[index];
    }

    pub fn getSolSets(&self) -> &Vec<SolSet> {
        return &self.solsets;
    }

    pub fn has_solset(&self, ssname: &str) -> bool {
        let result = &self.solsets.iter().find(|s| s.name == ssname);
        return match result {
            None => false,
            _ => true,
        };
    }
}

#[derive(Debug, Error)]
#[error("No soltab named {0} in h5parm!")]
struct MissingSoltabError(String);

#[derive(Debug)]
pub struct SolSet {
    pub name: String,
    pub soltabs: Vec<SolTab>,
}

impl SolSet {
    fn init(h5parm: &hdf5::File, name: String) -> Result<Self, Box<dyn std::error::Error>> {
        let _sts = h5parm
            .group(&name)
            .expect("Failed to read SolTabs.")
            .groups()
            .unwrap();
        let mut soltablist: Vec<SolTab> = vec![];
        for st in _sts.iter() {
            // VarLenAscii doesn't work, so we just read a long fixed-length string...
            // There's also some ASCII vs Unicode stuff, so try both.
            let st_type = match st
                .attr("TITLE")
                .expect("SolTab does not appear to have a type.")
                .read_scalar::<hdf5::types::FixedAscii<32>>()
            {
                Ok(f) => f.to_string().to_owned(),
                Err(_f) => "".to_string(),
            };

            let st_type = if st_type.is_empty() {
                match st
                    .attr("TITLE")
                    .expect("SolTab does not appear to have a type.")
                    .read_scalar::<hdf5::types::FixedUnicode<32>>()
                {
                    Ok(f) => f.as_str().to_owned(),
                    Err(_f) => "".to_string(),
                }
            } else {
                st_type.to_string()
            };
            let stname = st.name().rsplit_once("/").unwrap().1.to_string();
            let x = SolTab {
                name: stname,
                kind: match st_type.as_str() {
                    "amplitude" => SolTabKind::Amplitude,
                    "phase" => SolTabKind::Phase,
                    "clock" => SolTabKind::Clock,
                    "rotationmeasure" => SolTabKind::RotationMeasure,
                    _ => SolTabKind::Unknown,
                },
                is_fulljones: false,
                _solset: name.clone(),
                _h5parm: h5parm.clone(),
            };
            soltablist.push(x);
        }

        return Ok(SolSet {
            name: name,
            soltabs: soltablist,
        });
    }

    pub fn getSolTabs(&self) -> &Vec<SolTab> {
        return &self.soltabs;
    }

    pub fn getSolTab(&self, st_name: String) -> Result<&SolTab, anyhow::Error> {
        let index: i32 = if self.has_soltab(&st_name) {
            self.soltabs
                .iter()
                .position(|r| r.name == st_name)
                .unwrap()
                .try_into()
                .unwrap()
        } else {
            -1
        };
        if index < 0 {
            bail!(MissingSoltabError(st_name));
        }
        return Ok(&self.soltabs[index as usize]);
    }

    pub fn has_soltab(&self, stname: &str) -> bool {
        let result = &self.soltabs.iter().find(|s| s.name == stname);
        return match result {
            None => false,
            _ => true,
        };
    }
}

#[derive(Debug)]
pub struct SolTab {
    pub kind: SolTabKind,
    pub name: String,
    pub is_fulljones: bool,
    _solset: String,
    _h5parm: hdf5::File,
}

impl SolTab {
    /*
    pub fn new(&mut self) -> Self {
        SolTab {
            name: stname,
            kind: match st_type.as_str() {
                "amplitude" => SolTabKind::Amplitude,
                "phase" => SolTabKind::Phase,
                _ => SolTabKind::Unknown,
            },
            is_fulljones: false,
            _solset: name.clone(),
            _h5parm: h5parm.clone(),
        }
    }*/

    pub fn get_axes(&self) -> Vec<String> {
        let full_st_name = self.get_full_name();
        let _axes_string = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("val")
            .unwrap()
            .attr("AXES")
            .expect("SolTab is missing AXES attribute!")
            // Axes are time, ant, freq and optionally dir or pol.
            // This is a single comma-separated string, so max length 22 including commas.
            .read_scalar::<hdf5::types::FixedAscii<22>>()
            .unwrap();
        _axes_string.split(",").map(str::to_string).collect()
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_flagged_fraction(&self) -> Array1<f64> {
        let weights = self.get_weights();
        //let f = weights.iter().filter(|&n| *n == 0.0) / weights.sum();
        let zeros: Vec<_> = weights
        .iter()
        .filter_map(|&item| if item == 0.0 { Some(1) } else { Some(0) })
        .collect();
        let fraction = (zeros.iter().sum::<usize>() as f64) / (weights.len() as f64);
        array![fraction]
    }

    fn get_full_name(&self) -> String {
        format!("/{}/{}", self._solset, self.name)
    }

    pub fn get_type(&self) -> String {
        //&self.kind
        format!("{:?}", self.kind)
    }

    pub fn get_times(&self) -> Array1<f64> {
        let full_st_name = self.get_full_name();
        let st = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("time")
            .unwrap_or_else(|_err| {
                panic!(
                    "Failed to read times for SolTab {}",
                    stringify!(full_st_name)
                )
            });
        st.read_1d::<f64>().unwrap()
    }

    pub fn get_frequencies(&self) -> Array1<f64> {
        let full_st_name = self.get_full_name();
        let st = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("freq")
            .unwrap_or_else(|_err| {
                panic!(
                    "Failed to read frequencies for SolTab {}",
                    stringify!(full_st_name)
                )
            });
        st.read_1d::<f64>().unwrap()
    }

    pub fn get_antennas(&self) -> Array1<hdf5::types::FixedAscii<9>> {
        let full_st_name = self.get_full_name();
        let st = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("ant")
            .unwrap_or_else(|_err| {
                panic!(
                    "Failed to read antennas for SolTab {}",
                    stringify!(full_st_name)
                )
            });
        // Station names are at most 9 characters long, e.g. CS003HBA0, IE613HBA.
        st.read_1d::<hdf5::types::FixedAscii<9>>().unwrap()
    }

    pub fn get_directions(&self) -> Array1<hdf5::types::FixedAscii<128>> {
        let full_st_name = self.get_full_name();
        let st = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("dir")
            .unwrap_or_else(|_err| {
                panic!(
                    "Failed to read directions for SolTab {}",
                    stringify!(full_st_name)
                )
            });
        // Not sure what to do here. Surely 128 characters is fine?
        st.read_1d::<hdf5::types::FixedAscii<128>>().unwrap()
    }

    pub fn get_polarisations(&self) -> Array1<hdf5::types::FixedAscii<2>> {
        if !self.get_axes().contains(&"pol".to_string()) {
            array![hdf5::types::FixedAscii::<2>::from_ascii("").unwrap()]
        } else {
            let full_st_name = self.get_full_name();
            let st = self
                ._h5parm
                .group(&full_st_name)
                .unwrap()
                .dataset("pol")
                .unwrap_or_else(|_err| {
                    panic!(
                        "Failed to read polarisations for SolTab {}",
                        stringify!(full_st_name)
                    )
                });
            // Polarisations have at most 2 letters (usually), e.g. I, Q, U, V, XX, YY, RL, LR etc.
            st.read_1d::<hdf5::types::FixedAscii<2>>().unwrap()
        }
    }

    pub fn get_values(&self) -> ArrayD<f64> {
        let full_st_name = self.get_full_name();
        let st = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("val")
            .unwrap_or_else(|_err| {
                panic!(
                    "Failed to read values for SolTab {}",
                    stringify!(full_st_name)
                )
            });
        st.read_dyn::<f64>()
            .expect("Reading SolTab into array failed!")
    }

    pub fn get_weights(&self) -> ArrayD<f64> {
        let full_st_name = self.get_full_name();
        let st = self
            ._h5parm
            .group(&full_st_name)
            .unwrap()
            .dataset("weight")
            .unwrap_or_else(|_err| {
                panic!(
                    "Failed to read weights for SolTab {}",
                    stringify!(full_st_name)
                )
            });
        st.read_dyn::<f64>()
            .expect("Reading SolTab into array failed!")
    }
}

#[derive(Debug)]
pub enum SolTabKind {
    Amplitude,
    Clock,
    Phase,
    RotationMeasure,
    Unknown,
}
