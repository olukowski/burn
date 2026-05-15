use crate::{
    CubeRuntime,
    kernel::{into_contiguous, utils::address_type},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use cubecl::{calculate_cube_count_elemwise, prelude::*, std::tensor::layout::linear::LinearView};

#[cube(launch_unchecked, address_type = "dynamic")]
fn xnor_popcount_matmul_kernel<I: Int>(
    lhs: &LinearView<I>,
    rhs: &LinearView<I>,
    output: &mut LinearView<I, ReadWrite>,
    #[comptime] words: usize,
    #[comptime] out_features: usize,
    #[define(I)] _dtype: StorageType,
) {
    let out_pos = ABSOLUTE_POS;

    if !output.is_in_bounds(out_pos) {
        terminate!();
    }

    let batch = out_pos / out_features;
    let out_feature = out_pos % out_features;
    let mut total = I::new(0);

    for word in 0..words {
        let lhs_value = lhs[batch * words + word];
        let rhs_value = rhs[word * out_features + out_feature];
        total += I::cast_from((!(lhs_value ^ rhs_value)).count_ones());
    }

    output[out_pos] = total;
}

pub(crate) fn xnor_popcount_matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
) -> CubeTensor<R> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    assert_eq!(lhs_shape.num_dims(), 2);
    assert_eq!(rhs_shape.num_dims(), 2);
    let lhs_dims = lhs_shape.dims::<2>();
    let rhs_dims = rhs_shape.dims::<2>();
    assert_eq!(lhs_dims[1], rhs_dims[0]);

    let batch_size = lhs_dims[0];
    let words = lhs_dims[1];
    let out_features = rhs_dims[1];
    let output_shape = burn_backend::Shape::from([batch_size, out_features]);

    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);
    let output = empty_device_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        output_shape,
        lhs.dtype,
    );

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&lhs.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&lhs.client, num_elems, cube_dim);
    let dtype = lhs.dtype;

    unsafe {
        xnor_popcount_matmul_kernel::launch_unchecked::<R>(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(lhs, rhs, output),
            lhs.into_linear_view(),
            rhs.into_linear_view(),
            output.clone().into_linear_view(),
            words,
            out_features,
            dtype.into(),
        );
    }

    output
}
