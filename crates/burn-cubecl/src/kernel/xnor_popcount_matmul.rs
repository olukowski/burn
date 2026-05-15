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

#[cube(launch_unchecked, address_type = "dynamic")]
fn packed_attention_step_bits_kernel<I: Int>(
    query: &LinearView<I>,
    keys: &LinearView<I>,
    values: &LinearView<I>,
    bits_out: &mut LinearView<I, ReadWrite>,
    #[comptime] heads: usize,
    #[comptime] sequence: usize,
    #[comptime] head_words: usize,
    #[comptime] threshold: i64,
    #[define(I)] _dtype: StorageType,
) {
    let bit_pos = ABSOLUTE_POS;

    if !bits_out.is_in_bounds(bit_pos) {
        terminate!();
    }

    let out_pos = bit_pos / 32;
    let bit = bit_pos % 32;
    let word = out_pos % head_words;
    let head = (out_pos / head_words) % heads;
    let batch = out_pos / (heads * head_words);
    let query_base = (batch * heads + head) * head_words;
    let cache_base = ((batch * heads + head) * sequence) * head_words;
    let mut selected = I::new(0);
    let mut value_ones = I::new(0);

    for token in 0..sequence {
        let mut score = I::new(0);

        for score_word in 0..head_words {
            let query_value = query[query_base + score_word];
            let key_value = keys[cache_base + token * head_words + score_word];
            score += I::cast_from((!(query_value ^ key_value)).count_ones());
        }

        if score >= I::new(threshold) {
            selected += I::new(1);

            let value = values[cache_base + token * head_words + word];
            value_ones += (value >> I::cast_from(bit)) & I::new(1);
        }
    }

    let mut flag = I::new(0);
    if selected > I::new(0) && value_ones + value_ones >= selected {
        flag = I::new(1);
    }
    bits_out[bit_pos] = flag;
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn packed_attention_step_pack_kernel<I: Int>(
    bits: &LinearView<I>,
    output: &mut LinearView<I, ReadWrite>,
    #[define(I)] _dtype: StorageType,
) {
    let out_pos = ABSOLUTE_POS;

    if !output.is_in_bounds(out_pos) {
        terminate!();
    }

    let mut packed = I::new(0);

    #[unroll]
    for bit in 0..32 {
        if bits[out_pos * 32 + bit] > I::new(0) {
            let mask = comptime![if bit == 31 {
                i32::MIN as i64
            } else {
                1i64 << bit
            }];
            packed |= I::new(mask);
        }
    }

    output[out_pos] = packed;
}

pub(crate) fn packed_attention_step<R: CubeRuntime>(
    query: CubeTensor<R>,
    keys: CubeTensor<R>,
    values: CubeTensor<R>,
    threshold: i64,
) -> CubeTensor<R> {
    let query_shape = query.shape();
    let keys_shape = keys.shape();
    let values_shape = values.shape();

    assert_eq!(query_shape.num_dims(), 3);
    assert_eq!(keys_shape.num_dims(), 4);
    assert_eq!(values_shape.num_dims(), 4);

    let query_dims = query_shape.dims::<3>();
    let keys_dims = keys_shape.dims::<4>();
    let values_dims = values_shape.dims::<4>();

    assert_eq!(query_dims[0], keys_dims[0]);
    assert_eq!(query_dims[0], values_dims[0]);
    assert_eq!(query_dims[1], keys_dims[1]);
    assert_eq!(query_dims[1], values_dims[1]);
    assert_eq!(query_dims[2], keys_dims[3]);
    assert_eq!(query_dims[2], values_dims[3]);
    assert_eq!(keys_dims[2], values_dims[2]);

    let heads = query_dims[1];
    let sequence = keys_dims[2];
    let head_words = query_dims[2];
    let output_shape = burn_backend::Shape::from(query_dims);
    let bits_shape = burn_backend::Shape::from([query_dims[0], query_dims[1], query_dims[2], 32]);

    let query = into_contiguous(query);
    let keys = into_contiguous(keys);
    let values = into_contiguous(values);
    let bits = empty_device_dtype(
        query.client.clone(),
        query.device.clone(),
        bits_shape,
        query.dtype,
    );
    let output = empty_device_dtype(
        query.client.clone(),
        query.device.clone(),
        output_shape,
        query.dtype,
    );

    let bit_elems = bits.meta.num_elements();
    let bit_cube_dim = CubeDim::new(&query.client, bit_elems);
    let bit_cube_count = calculate_cube_count_elemwise(&query.client, bit_elems, bit_cube_dim);
    let dtype = query.dtype;

    unsafe {
        packed_attention_step_bits_kernel::launch_unchecked::<R>(
            &bits.client,
            bit_cube_count,
            bit_cube_dim,
            address_type!(query, keys, values, bits),
            query.into_linear_view(),
            keys.into_linear_view(),
            values.into_linear_view(),
            bits.clone().into_linear_view(),
            heads,
            sequence,
            head_words,
            threshold,
            dtype.into(),
        );
    }

    let num_elems = output.meta.num_elements();
    let cube_dim = CubeDim::new(&output.client, num_elems);
    let cube_count = calculate_cube_count_elemwise(&output.client, num_elems, cube_dim);

    unsafe {
        packed_attention_step_pack_kernel::launch_unchecked::<R>(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(bits, output),
            bits.into_linear_view(),
            output.clone().into_linear_view(),
            dtype.into(),
        );
    }

    output
}
