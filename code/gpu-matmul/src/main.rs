
use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
use std::time::Instant;

// Convert (row, col) into an offset in row-major memory.
#[inline]
fn index(row: usize, col: usize, width: usize) -> usize {
    row * width + col
}

#[cuda_module]
mod kernels {
    use super::*;

    // C = A x B, with A: M x K, B: K x N and C: M x N.
    #[kernel]
    pub fn matmul(m: u32, n: u32, k: u32, a: &[f32], b: &[f32], c: &mut [f32]) {
        let (m, n, k) = (m as usize, n as usize, k as usize);

        // Row-major offset. One helper for all three matrices; the width is
        // its last argument because A is K wide while B and C are N wide.
        let index = |row: usize, col: usize, width: usize| row * width + col;

        // Global coordinates of this GPU thread.
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();

        // Threads outside C have nothing to write.
        if row >= m || col >= n {
            return;
        }

        // One thread calculates one element of C.
        let mut sum = 0.0f32;

        for i in 0..k {
            sum += a[index(row, i, k)] * b[index(i, col, n)];
        }

        c[index(row, col, n)] = sum;
    }
}

type MayBeError = Box<dyn std::error::Error>;

fn main() -> Result<(), MayBeError> {
    // 1. Initialize GPU.
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // 2. C = A x B, with A: M x K and B: K x N.
    //    Block edge: the launch runs block x block threads per block.
    let (m, k, n) = (128, 64, 96);
    let block = 16u32;

    let (a_host, b_host) = inputs(m, k, n);
    let a_gpu = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_gpu = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_gpu = DeviceBuffer::<f32>::zeroed(&stream, m * n)?;

    // 3. Launch.
    let module = kernels::load(&ctx)?;

    let start = Instant::now();
    launch(&module, &stream, (m, k, n), block, &a_gpu, &b_gpu, &mut c_gpu)?;
    println!("GPU Kernel Execution took : {:.3} ms", start.elapsed().as_secs_f64() * 1e3);

    // 4. Bring the result back and show it.
    let result = c_gpu.to_host_vec(&stream)?;
    println!("It took total : {:.3} ms including copying the data back to the host.", start.elapsed().as_secs_f64() * 1e3);

    println!("C = A x B, with A: {m}x{k}, B: {k}x{n}, C: {m}x{n}");
    print_corner(&result, n, 4);

    // 5. Check every element against the CPU.
    verify(&a_host, &b_host, &result, (m, k, n));
    println!("PASSED: GPU matches CPU on all {} elements", m * n);

    Ok(())
}

// A (M x K) and B (K x N), row-major and generated deterministically.
//
// Every value is a small integer in [-5, 5], so all products and partial sums
// are exactly representable and the CPU check is exact, not approximate.
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

// 2D launch over C: X = columns (N), Y = rows (M), Z = 1.
fn launch(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    (m, k, n): (usize, usize, usize),
    block: u32,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &mut DeviceBuffer<f32>,
) -> Result<(), MayBeError> {
    let config = LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(block),
            (m as u32).div_ceil(block),
            1,
        ),
        block_dim: (block, block, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - The launch uses a 2D grid over the M x N output.
    // - Each thread has a unique (row, col), bounds-checked in the kernel.
    // - A is M*K, B is K*N and C is M*N elements.
    unsafe {
        module.matmul(
            stream,
            config,
            m as u32,
            n as u32,
            k as u32,
            a,
            b,
            c,
        )?;
    }

    Ok(())
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
