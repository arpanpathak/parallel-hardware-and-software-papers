use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{kernel, thread, SharedArray};
use cuda_host::cuda_module;
use std::time::Instant;

/// Edge length of a tile. Shared arrays are sized at compile time,
/// so this is the one constant; every matrix dimension stays dynamic.
const TILE: usize = 16;

// Convert (row, col) into an offset in row-major memory.
#[inline]
fn index(row: usize, col: usize, width: usize) -> usize {
    row * width + col
}

#[cuda_module]
mod kernels {
    use super::*;

    // C = A x B, with A: M x K, B: K x N and C: M x N.
    // One thread computes one element of C, reading straight from global memory.
    #[kernel]
    pub fn matmul_naive(m: u32, n: u32, k: u32, a: &[f32], b: &[f32], c: &mut [f32]) {
        let (m, n, k) = (m as usize, n as usize, k as usize);
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        if row >= m || col >= n {
            return;
        }

        let mut sum = 0.0f32;
        for i in 0..k {
            sum += a[index(row, i, k)] * b[index(i, col, n)];
        }
        c[index(row, col, n)] = sum;
    }

    // Same result, but each block stages TILE x TILE pieces of A and B
    // in shared memory and every thread of the block reuses them.
    #[kernel]
    pub fn matmul_tiled(m: u32, n: u32, k: u32, a: &[f32], b: &[f32], c: &mut [f32]) {
        static mut TILE_A: SharedArray<f32, { TILE * TILE }> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, { TILE * TILE }> = SharedArray::UNINIT;

        let (m, n, k) = (m as usize, n as usize, k as usize);
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let mut sum = 0.0f32;
        for t in 0..k.div_ceil(TILE) {
            let a_col = t * TILE + tx;
            let b_row = t * TILE + ty;

            // Load: each thread copies one element of A and one of B.
            // Positions past the edge of a matrix are filled with zero.
            // SAFETY: each thread owns exactly one slot (ty, tx) in each
            // shared tile, and the block is TILE x TILE threads. The reads
            // from a and b are guarded by the bounds checks.
            unsafe {
                TILE_A[index(ty, tx, TILE)] =
                    if row < m && a_col < k { a[index(row, a_col, k)] } else { 0.0 };
                TILE_B[index(ty, tx, TILE)] =
                    if b_row < k && col < n { b[index(b_row, col, n)] } else { 0.0 };
            }
            thread::sync_threads(); // wait until both tiles are full

            // Compute: one partial dot product, entirely from shared memory.
            // SAFETY: the barrier above completes both tiles before any thread
            // reads them, and no thread writes them until the barrier below.
            for i in 0..TILE {
                unsafe {
                    sum += TILE_A[index(ty, i, TILE)] * TILE_B[index(i, tx, TILE)];
                }
            }
            thread::sync_threads(); // wait until every thread has read the tiles
        }

        if row < m && col < n {
            c[index(row, col, n)] = sum;
        }
    }
}

type MayBeError = Box<dyn std::error::Error>;

fn main() -> Result<(), MayBeError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    // Any sizes work; none of them has to be a multiple of TILE.
    let (m, k, n) = (128, 64, 96);

    let (a_host, b_host) = inputs(m, k, n);
    let a_gpu = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_gpu = DeviceBuffer::from_host(&stream, &b_host)?;

    // A 2D grid over C: x runs along columns (N), y along rows (M).
    let side = TILE as u32;
    let config = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(side), (m as u32).div_ceil(side), 1),
        block_dim: (side, side, 1),
        shared_mem_bytes: 0,
    };

    for tiled in [false, true] {
        let mut c_gpu = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;
        let start = Instant::now();

        // SAFETY: the grid covers C, each thread writes only its own (row, col),
        // and the buffers hold M*K, K*N and M*N elements.
        unsafe {
            if tiled {
                module.matmul_tiled(
                    &stream, config, m as u32, n as u32, k as u32, &a_gpu, &b_gpu, &mut c_gpu,
                )?;
            } else {
                module.matmul_naive(
                    &stream, config, m as u32, n as u32, k as u32, &a_gpu, &b_gpu, &mut c_gpu,
                )?;
            }
        }

        // Copying back waits for the kernel, so this time includes its execution.
        let result = c_gpu.to_host_vec(&stream)?;
        let ms = start.elapsed().as_secs_f64() * 1e3;
        let label = if tiled { "tiled" } else { "naive" };
        println!("{label}: {ms:.3} ms, kernel plus copy back");

        verify(&a_host, &b_host, &result, (m, k, n));
        println!("{label}: GPU matches CPU on all {} elements", m * n);
        print_corner(&result, n, 4);
    }

    Ok(())
}

// A (M x K) and B (K x N), row-major and generated deterministically.
// Every value is a small integer in [-5, 5], so all products and partial
// sums are exactly representable and the CPU check is exact.
fn inputs(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let generate = |rows: usize, cols: usize, seed: usize| -> Vec<f32> {
        (0..rows * cols)
            .map(|i| {
                let (row, col) = (i / cols, i % cols);
                ((row * 13 + col * 7 + seed) % 11) as f32 - 5.0
            })
            .collect()
    };
    (generate(m, k, 0), generate(k, n, 5))
}

// Check every element against a plain CPU implementation.
fn verify(a: &[f32], b: &[f32], c: &[f32], (m, k, n): (usize, usize, usize)) {
    for row in 0..m {
        for col in 0..n {
            let expected: f32 = (0..k)
                .map(|i| a[index(row, i, k)] * b[index(i, col, n)])
                .sum();
            let actual = c[index(row, col, n)];

            assert!(
                (actual - expected).abs() < 1e-5,
                "Mismatch at ({row}, {col}): got {actual}, want {expected}"
            );
        }
    }
}

// The full C is too wide to read in a terminal, so show its top-left corner.
fn print_corner(values: &[f32], width: usize, size: usize) {
    let size = size.min(values.len() / width).min(width);

    println!("C[0..{size}][0..{size}]:");
    for row in 0..size {
        let start = row * width;
        let cells: Vec<String> = values[start..start + size]
            .iter()
            .map(|value| format!("{value:>8.1}"))
            .collect();
        println!("  {}", cells.join(" "));
    }
}
