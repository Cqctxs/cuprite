// Code that the CPU programs and the GPU kernels both include, so they
// compute the same thing from one source. Must stay `no_std`-compatible.

/// Barrel (k > 0) or pincushion (k < 0) lens distortion about `(cx, cy)`:
/// the mapping an application passes to `imageproc::warp_with`.
pub fn barrel(x: f32, y: f32, cx: f32, cy: f32, k: f32) -> (f32, f32) {
    let dx = x - cx;
    let dy = y - cy;
    let r2 = (dx * dx + dy * dy) * k;
    let factor = 1.0 + r2;
    (cx + dx * factor, cy + dy * factor)
}

// SDF scenes for `sdfu`: the numbers, shared so the CPU scenes and the GPU
// ports can't drift apart.

/// A machined part: a rounded plate with a boss blended on, a bore through
/// it, and a slot across it.
pub const PART_BODY: [f32; 3] = [0.9, 0.6, 0.35];
pub const PART_ROUNDING: f32 = 0.08;
pub const PART_BOSS_CENTER: [f32; 3] = [0.6, 0.0, 0.3];
pub const PART_BOSS_RADIUS: f32 = 0.45;
pub const PART_BLEND: f32 = 0.1;
/// Radius and half-height of the bore along z.
pub const PART_BORE: [f32; 2] = [0.3, 0.5];
pub const PART_SLOT_OFFSET: [f32; 3] = [0.0, 0.35, 0.0];
pub const PART_SLOT_RADIUS: f32 = 0.12;

/// Metaballs: spheres on a helix, smoothly unioned in order.
pub const BALLS: [[f32; 3]; 8] = [
    [0.7, -0.63, 0.0],
    [0.49, -0.45, 0.5],
    [0.0, -0.27, 0.7],
    [-0.49, -0.09, 0.5],
    [-0.7, 0.09, 0.0],
    [-0.49, 0.27, -0.5],
    [0.0, 0.45, -0.7],
    [0.49, 0.63, -0.5],
];
pub const BALL_RADIUS: f32 = 0.35;
pub const BALL_BLEND: f32 = 0.25;

/// A wireframe cube: a capsule along each edge, unioned with a torus.
pub const WIRE_EDGES: [[[f32; 3]; 2]; 12] = [
    [[-0.7, -0.7, -0.7], [0.7, -0.7, -0.7]],
    [[-0.7, 0.7, -0.7], [0.7, 0.7, -0.7]],
    [[-0.7, -0.7, 0.7], [0.7, -0.7, 0.7]],
    [[-0.7, 0.7, 0.7], [0.7, 0.7, 0.7]],
    [[-0.7, -0.7, -0.7], [-0.7, 0.7, -0.7]],
    [[0.7, -0.7, -0.7], [0.7, 0.7, -0.7]],
    [[-0.7, -0.7, 0.7], [-0.7, 0.7, 0.7]],
    [[0.7, -0.7, 0.7], [0.7, 0.7, 0.7]],
    [[-0.7, -0.7, -0.7], [-0.7, -0.7, 0.7]],
    [[0.7, -0.7, -0.7], [0.7, -0.7, 0.7]],
    [[-0.7, 0.7, -0.7], [-0.7, 0.7, 0.7]],
    [[0.7, 0.7, -0.7], [0.7, 0.7, 0.7]],
];
pub const WIRE_THICKNESS: f32 = 0.05;
/// `Torus::new(radius, thickness)`: tube radius, then ring radius.
pub const WIRE_TORUS: [f32; 2] = [0.1, 0.45];

/// How many leading bits of a digest are zero, at most 255: the score a
/// proof-of-work or seed search keeps for each candidate.
pub fn leading_zero_bits(digest: &[u8]) -> u8 {
    let mut bits = 0u32;
    for &byte in digest {
        if byte != 0 {
            bits += byte.leading_zeros();
            break;
        }
        bits += 8;
    }
    bits.min(255) as u8
}

// SHA-2 constants, copied from sha2 0.10.9's src/consts.rs.
pub const K32: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
pub const K64: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];
pub const H256_256: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
pub const H512_512: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];
