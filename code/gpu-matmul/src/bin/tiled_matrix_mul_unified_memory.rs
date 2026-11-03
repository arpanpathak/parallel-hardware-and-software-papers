// Tiled C = A x B where the GPU reads A and B straight from host memory.
// On Jetson AGX Thor, CPU and GPU share 1 physical memory, so A and B are
// never copied. Only C (1 MiB) is copied back.

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{kernel, thread, SharedArray};
use cuda_host::{cuda_module, load_kernel_module};
use std::time::Instant;

const TILE: usize = 16;

#[inline]
fn index(row: usize, col: usize, width: usize) -> usize {
    row * width + col
}

#[cuda_module]
mod kernels {
    use super::*;

    // The tiled kernel, except A and B are closures over host slices.
    #[kernel]
    pub fn matmul_unified<FA: Fn(usize) -> f32 + Copy, FB: Fn(usize) -> f32 + Copy>(
        m: u32, n: u32, k: u32, a: FA, b: FB, c: &mut [f32],
    ) {
        static mut TILE_A: SharedArray<f32, { TILE * TILE }> = SharedArray::UNINIT;
        static mut TILE_B: SharedArray<f32, { TILE * TILE }> = SharedArray::UNINIT;

        let (m, n, k) = (m as usize, n as usize, k as usize);
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let tx = thread::threadIdx_x() as usize;
        let ty = thread::threadIdx_y() as usize;

        let mut sum = 0.0f32;
        for t in 0..k.div_ceil(TILE) {
            let (a_col, b_row) = (t * TILE + tx, t * TILE + ty);
            // SAFETY: each thread owns exactly one slot (ty, tx) in each
            // shared tile. The reads through the host-slice closures are
            // guarded by the bounds checks.
            unsafe {
                TILE_A[index(ty, tx, TILE)] = if row < m && a_col < k { a(index(row, a_col, k)) } else { 0.0 };
                TILE_B[index(ty, tx, TILE)] = if b_row < k && col < n { b(index(b_row, col, n)) } else { 0.0 };
            }
            thread::sync_threads(); // wait until both tiles are full

            // SAFETY: the barrier above completes both tiles before any thread
            // reads them, and no thread writes them until the barrier below.
            for i in 0..TILE {
                unsafe { sum += TILE_A[index(ty, i, TILE)] * TILE_B[index(i, tx, TILE)]; }
            }
            thread::sync_threads(); // wait until every thread has read the tiles
        }

        if row < m && col < n {
            c[index(row, col, n)] = sum;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // A kernel that takes closures is loaded by name, then wrapped.
    let module = load_kernel_module(&ctx, "tiled_matrix_mul_unified_memory")
        .map_err(|e| format!("load: {e}"))?;
    let module = kernels::from_module(module).map_err(|e| format!("from_module: {e}"))?;

    let n = 512; // square: M = K = N
    let a_host: Vec<f32> = (0..n * n).map(|i| ((i * 7) % 11) as f32 - 5.0).collect();
    let b_host: Vec<f32> = (0..n * n).map(|i| ((i * 13 + 5) % 11) as f32 - 5.0).collect();
    let (a, b): (&[f32], &[f32]) = (&a_host, &b_host);

    let side = TILE as u32;
    let config = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(side), (n as u32).div_ceil(side), 1),
        block_dim: (side, side, 1),
        shared_mem_bytes: 0,
    };

    let run = || -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut c_gpu = DeviceBuffer::<f32>::zeroed(&stream, n * n)?;
        // SAFETY: the grid covers C, the host-slice closures stay inside the
        // n x n inputs, and each thread writes only its own (row, col).
        unsafe {
            module.matmul_unified::<_, _>(
                stream.as_ref(), config, n as u32, n as u32, n as u32,
                move |i: usize| a[i], move |i: usize| b[i], &mut c_gpu,
            )?;
        }
        Ok(c_gpu.to_host_vec(&stream)?) // waits for the kernel
    };

    run()?; // warm-up
    let start = Instant::now();
    let c = run()?;
    println!("unified: {:.3} ms end to end (kernel + copy C back), {n}x{n}", start.elapsed().as_secs_f64() * 1e3);

    for (row, col) in [(0, 0), (1, 2), (n / 2, n / 3), (n - 1, n - 1)] {
        let want: f32 = (0..n).map(|s| a[row * n + s] * b[s * n + col]).sum();
        assert_eq!(c[row * n + col], want, "mismatch at ({row}, {col})");
    }
    println!("PASSED");
    Ok(())
}
