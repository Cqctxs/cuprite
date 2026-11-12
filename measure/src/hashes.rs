//! Hashing many candidates with the `sha2` crate: `prefix ‖ nonce` for a
//! range of nonces, as proof-of-work and seed searches do. Each candidate
//! keeps its digest's leading zero bits, or its whole digest.

use cuprite_memory::{Arg, Grid};
use rayon::prelude::*;
use sha2::{Digest, Sha256, Sha512};

use crate::{leading_zero_bits, Case, Gpu, Output};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Algorithm {
    Sha256,
    Sha512,
}

impl Algorithm {
    fn block(self) -> usize {
        match self {
            Algorithm::Sha256 => 64,
            Algorithm::Sha512 => 128,
        }
    }

    fn digest_len(self) -> usize {
        match self {
            Algorithm::Sha256 => 32,
            Algorithm::Sha512 => 64,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HashCase {
    pub algorithm: Algorithm,
    pub prefix: Vec<u8>,
    pub first: u64,
    pub count: u64,
    /// Whole digests instead of one score byte per candidate.
    pub digests: bool,
}

impl HashCase {
    /// A search over `count` nonces after a prefix of `prefix_len` bytes.
    pub fn new(algorithm: Algorithm, prefix_len: usize, count: u64) -> Self {
        Self {
            algorithm,
            prefix: (0..prefix_len).map(|i| (i * 37 + 11) as u8).collect(),
            first: 0x0123_4567_89ab_cdef,
            count,
            digests: false,
        }
    }

    /// The crate, for one candidate.
    fn hash(&self, nonce: u64, out: &mut [u8]) {
        let nonce = nonce.to_le_bytes();
        let digest: Vec<u8> = match self.algorithm {
            Algorithm::Sha256 => Sha256::new()
                .chain_update(&self.prefix)
                .chain_update(nonce)
                .finalize()
                .to_vec(),
            Algorithm::Sha512 => Sha512::new()
                .chain_update(&self.prefix)
                .chain_update(nonce)
                .finalize()
                .to_vec(),
        };
        if self.digests {
            out.copy_from_slice(&digest);
        } else {
            out[0] = leading_zero_bits(&digest);
        }
    }

    fn out_len(&self) -> usize {
        if self.digests {
            self.algorithm.digest_len()
        } else {
            1
        }
    }

    /// The padded message with a zero nonce: the message, `0x80`, zeros,
    /// and the bit length, big-endian, in the last 8 (SHA-256) or 16
    /// (SHA-512) bytes of the last block.
    fn template(&self) -> Vec<u8> {
        let block = self.algorithm.block();
        let length_bytes = block / 8;
        let len = self.prefix.len() + 8;
        let mut message = self.prefix.clone();
        message.extend([0; 8]);
        message.push(0x80);
        while message.len() % block != block - length_bytes {
            message.push(0);
        }
        let bits = (len as u128) * 8;
        message.extend(&bits.to_be_bytes()[16 - length_bytes..]);
        message
    }
}

impl Case for HashCase {
    fn name(&self) -> String {
        let algorithm = match self.algorithm {
            Algorithm::Sha256 => "sha-256",
            Algorithm::Sha512 => "sha-512",
        };
        let what = if self.digests { "digests" } else { "search" };
        let blocks = self.template().len() / self.algorithm.block();
        format!(
            "{algorithm} {what}, {}-byte messages ({blocks} block{})",
            self.prefix.len() + 8,
            if blocks == 1 { "" } else { "s" }
        )
    }

    fn size(&self) -> (usize, usize) {
        (self.count as usize, 1)
    }

    fn shape(&self) -> String {
        format!("{:.1}M hashes", self.count as f64 / 1e6)
    }

    fn kernel_name(&self) -> &'static str {
        match self.algorithm {
            Algorithm::Sha256 => "sha256_candidates",
            Algorithm::Sha512 => "sha512_candidates",
        }
    }

    fn cpu(&self) -> Output {
        let width = self.out_len();
        let mut out = vec![0u8; self.count as usize * width];
        for (i, slot) in out.chunks_exact_mut(width).enumerate() {
            self.hash(self.first.wrapping_add(i as u64), slot);
        }
        Output::Bytes(out)
    }

    fn cpu_parallel(&self) -> Output {
        let width = self.out_len();
        let mut out = vec![0u8; self.count as usize * width];
        out.par_chunks_exact_mut(width)
            .enumerate()
            .for_each(|(i, slot)| self.hash(self.first.wrapping_add(i as u64), slot));
        Output::Bytes(out)
    }

    fn gpu(&self, gpu: &mut Gpu) -> Output {
        let template = self.template();
        let kernel = gpu.kernel(self.kernel_name());
        let mut out = vec![0u8; self.count as usize * self.out_len()];
        let mut job = gpu.memory.job().unwrap();
        let template_buffer = job.input(&template).unwrap();
        let output = job.output(&mut out).unwrap();
        let args = [
            Arg::Buffer(output),
            Arg::Buffer(template_buffer),
            Arg::U32((template.len() / self.algorithm.block()) as u32),
            Arg::U32(self.prefix.len() as u32),
            Arg::U64(self.first),
            Arg::U64(self.count),
            Arg::U32(u32::from(self.digests)),
        ];
        job.launch(kernel, Grid::Linear(self.count), &args).unwrap();
        job.finish().unwrap();
        Output::Bytes(out)
    }
}
