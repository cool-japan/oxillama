//! Sign-mask tables shared across IQ2/IQ3 codebooks.

/// Per-weight sign bit masks (8 entries, one per weight within a group of 8).
///
/// `KMASK_IQ2XS[j]` is a power-of-two mask selecting bit `j` from a sign byte.
/// A weight is negated when `sign_byte & KMASK_IQ2XS[j] != 0`.
pub const KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

/// 7-bit-index → sign-byte lookup table (128 entries).
///
/// Given a 7-bit value `v` extracted from the packed block data,
/// `KSIGNS_IQ2XS[v]` is a byte where bit `j` being 1 means weight `j`
/// in the group of 8 should be negated.
pub const KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];
