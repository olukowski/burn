use crate::{
    kernel::{into_contiguous, utils::address_type},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
    CubeRuntime,
};
use burn_backend::TensorMetadata;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
fn pack_bits_kernel<I: Int>(
    input: &LinearView<I>,
    output: &mut LinearView<I, ReadWrite>,
    #[define(I)] _dtype: StorageType,
) {
    let group = ABSOLUTE_POS;

    if !output.is_in_bounds(group) {
        terminate!();
    }

    let mut packed = I::new(0);

    #[unroll]
    for bit in 0..32 {
        packed |= input[group * 32 + bit] << I::new(comptime![bit as i64]);
    }

    output[group] = packed;
}

pub(crate) fn pack_bits<R: CubeRuntime>(input: CubeTensor<R>) -> CubeTensor<R> {
    let input_shape = input.shape();

    let rank = input_shape.num_dims();
    assert!(rank >= 2, "pack_bits input rank must be at least 2");
    assert_eq!(
        input_shape[rank - 1],
        32,
        "pack_bits input last dimension must be 32"
    );

    let output_shape = input_shape
        .iter()
        .copied()
        .take(rank - 1)
        .collect::<burn_backend::Shape>();
    let input = into_contiguous(input);
    let output = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        output_shape,
        input.dtype,
    );

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&input.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elems, cube_dim);
    let dtype = input.dtype;

    unsafe {
        pack_bits_kernel::launch_unchecked::<R>(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(input, output),
            input.into_linear_view(),
            output.clone().into_linear_view(),
            dtype.into(),
        );
    }

    output
}
